# Complete updated Sudoku solver training script for Colab
# - Streams from CSV with tf.data (no giant NumPy arrays)
# - Mixed precision for training (optional), but we cast back to float32 before TFLite conversion
# - Post-processes NN probabilities with a light constraint solver (guaranteed legal grids)
# - Streaming evaluation (no big vstack)
# - Same model architecture and callbacks

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt

# ---- Mount Drive (Colab) ----
from google.colab import drive
drive.mount('/content/drive')

# ---- Path to CSV ----
CSV_PATH = '/content/drive/MyDrive/Machine Learning CSV/Sudoku Solver/sudoku.csv'

# ---- Quick check for columns and basic CSV sanity ----
df_head = pd.read_csv(CSV_PATH, nrows=5)
assert 'quizzes' in df_head.columns and 'solutions' in df_head.columns, \
    "CSV must contain 'quizzes' and 'solutions' columns."

# Optional: check that each quizzes/solutions string is 81 characters (common format)
bad_rows = df_head[ (df_head['quizzes'].str.len() != 81) | (df_head['solutions'].str.len() != 81) ]
if len(bad_rows) > 0:
    print("Warning: some first rows don't have 81-character quizzes/solutions. Inspect CSV.")
else:
    print("Quick CSV head check passed (first rows look 81 chars).")

    # ---- tf.data parsing function (accepts '.' or '0' for blanks) ----
def parse_row(quiz_str, sol_str):
    # quiz_str, sol_str are tf.string tensors of the full 81-character string
    # quizzes: accept '.' or '0' as blanks, digits '1'..'9' for filled
    # split into single-character strings
    qs = tf.strings.bytes_split(quiz_str)  # shape (81,)
    # treat '.' as '0' (empty)
    qs = tf.where(tf.equal(qs, b'.'), b'0', qs)
    qnums = tf.strings.to_number(qs, out_type=tf.float32)  # 0..9
    qnums = tf.reshape(qnums, [9, 9, 1]) / 9.0            # scale to 0..1 for model input

    # solutions: expected '1'..'9' for each cell (no blanks)
    ss = tf.strings.bytes_split(sol_str)
    snums = tf.strings.to_number(ss, out_type=tf.int32)    # 1..9
    snums = tf.reshape(snums, [9, 9]) - 1                 # 0..8
    sol_onehot = tf.one_hot(snums, depth=9, dtype=tf.float32)  # [9,9,9]

    return qnums, sol_onehot

# ---- Dataset from CSV (streaming) ----
raw_ds = tf.data.experimental.CsvDataset(
    CSV_PATH, [tf.string, tf.string], header=True
).map(parse_row, num_parallel_calls=tf.data.AUTOTUNE)

# ---- Train/val split sizes ----
# compute total rows (fast rough method)
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    total_rows = sum(1 for _ in f) - 1  # minus header
train_size = int(0.8 * total_rows)
val_size = total_rows - train_size
print(f"Total rows: {total_rows}, train: {train_size}, val: {val_size}")

BATCH_SIZE = 64  # increase if you have RAM/GPU room (e.g., 128/256)
SHUFFLE_BUFFER = 2048

# Build datasets (train repeated for steps_per_epoch, val repeated for training validation)
train_ds = raw_ds.take(train_size) \
    .shuffle(SHUFFLE_BUFFER) \
    .repeat() \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = raw_ds.skip(train_size) \
    .repeat() \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# For final streaming evaluation (single pass, not repeated), create eval_ds (no .repeat())
eval_ds = raw_ds.skip(train_size).batch(BATCH_SIZE)

steps_per_epoch = max(1, train_size // BATCH_SIZE)
validation_steps = max(1, val_size // BATCH_SIZE)

# ---- Mixed precision (modern API) ----
# We enable mixed precision for faster training if supported, but we'll cast back to float32 before TFLite.
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled:", mixed_precision.global_policy())
except Exception as e:
    print("Mixed precision not enabled (continuing without it):", e)

    # ---- Model ----n
num_classes = 9

def build_model(input_shape=(9,9,1), dropout_rate=0.2):
    inp = Input(shape=input_shape, name='sudoku_input')
    x = Conv2D(64, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(dropout_rate)(x)
    x = Conv2D(num_classes, (1,1), padding='same')(x)
    # keep outputs float32 for numerical stability; model dtype may be float16 internally
    x = Activation('softmax', dtype='float32')(x)
    return Model(inputs=inp, outputs=x, name='sudoku_conv')

model = build_model()
model.summary()


# ---- Compile ----
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Add categorical accuracy metric for training monitoring (cell-level)
model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name='cell_acc')])

# ---- Callbacks ----
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
]

# ---- Train ----
EPOCHS = 10  # change as needed
history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---- Light constraint solver that uses model probabilities to produce a legal Sudoku ----
def solve_with_probs(probs, puzzle_clues):
    """
    probs: numpy array shape (9,9,9) probabilities for digits 1..9
    puzzle_clues: numpy array shape (9,9) integers 0..9 (0 = empty, 1..9 = clue)
    Returns solved grid (9,9) with digits 1..9 or None if no solution found.
    """
    probs = probs.copy()
    grid = puzzle_clues.copy().astype(int)

    # helpers: used sets
    rows_used = [set(grid[r, grid[r] > 0].tolist()) for r in range(9)]
    cols_used = [set(grid[:, c][grid[:, c] > 0].tolist()) for c in range(9)]
    boxes_used = []
    for br in range(3):
        for bc in range(3):
            r0, c0 = br*3, bc*3
            box = grid[r0:r0+3, c0:c0+3].flatten()
            boxes_used.append(set(box[box > 0].tolist()))

    # Precompute descending candidate order for each cell (digits 1..9)
    candidate_order = {}
    for r in range(9):
        for c in range(9):
            # argsort descending, +1 to convert 0..8 indices to digits 1..9
            order = np.argsort(-probs[r, c, :]) + 1
            candidate_order[(r, c)] = order.tolist()

    # Backtracking using dynamic MRV (choose empty cell with fewest viable candidates)
    def get_candidates(r, c):
        b = (r//3)*3 + (c//3)
        cand = [d for d in candidate_order[(r, c)]
                if (d not in rows_used[r]) and (d not in cols_used[c]) and (d not in boxes_used[b])]
        return cand

    # find all empty cells initially
    def find_empty_cells():
        return [(r, c) for r in range(9) for c in range(9) if grid[r, c] == 0]

    def backtrack():
        empties = find_empty_cells()
        if not empties:
            return True
        # choose cell with minimum remaining candidates
        best_rc = None
        best_cands = None
        best_len = 999
        for (r, c) in empties:
            cands = get_candidates(r, c)
            if len(cands) == 0:
                return False  # dead end
            if len(cands) < best_len:
                best_len = len(cands)
                best_rc = (r, c)
                best_cands = cands
        r, c = best_rc
        b = (r//3)*3 + (c//3)
        for d in best_cands:
            # place
            grid[r, c] = d
            rows_used[r].add(d)
            cols_used[c].add(d)
            boxes_used[b].add(d)
            if backtrack():
                return True
            # undo
            grid[r, c] = 0
            rows_used[r].remove(d)
            cols_used[c].remove(d)
            boxes_used[b].remove(d)
        return False

    solved = backtrack()
    return grid if solved else None

# ---- Evaluation (streaming, using the legal solver) ----
total_correct = 0
total_cells = 0
total_exact = 0
total_puzzles = 0

for batch_x, batch_y in eval_ds:  # eval_ds is a single-pass dataset (no repeat)
    # batch_x: [B,9,9,1], batch_y: [B,9,9,9] (one-hot)
    probs_batch = model.predict(batch_x, verbose=0)  # (B,9,9,9) as float32
    # convert batch_x to original clues (0..9)
    batch_x_np = batch_x.numpy()
    batch_y_np = batch_y.numpy()
    B = batch_x_np.shape[0]
    for i in range(B):
        probs_i = probs_batch[i]  # (9,9,9)
        # reconstruct clues (round to nearest int, because we scaled by /9.0 during parsing)
        puzzle_clues = np.rint(batch_x_np[i, :, :, 0] * 9.0).astype(int)  # 0..9
        # ensure clues are in 0..9
        puzzle_clues = np.clip(puzzle_clues, 0, 9)

        # try to solve legally using probs
        solved = solve_with_probs(probs_i, puzzle_clues)
        if solved is None:
            # fallback: greedy argmax (may be illegal), but this is rare — you could also try time-limited deeper search
            greedy = np.argmax(probs_i, axis=-1) + 1
            final = greedy
        else:
            final = solved  # legal 1..9 grid

        # true digits
        true_digits = np.argmax(batch_y_np[i], axis=-1) + 1  # shape (9,9) values 1..9

        matches = (final == true_digits)
        total_correct += matches.sum()
        total_cells += matches.size
        total_exact += int(np.all(matches))
        total_puzzles += 1

# final metrics
cell_acc = total_correct / total_cells if total_cells > 0 else 0.0
puzzle_acc = total_exact / total_puzzles if total_puzzles > 0 else 0.0
print(f"Validation digit accuracy (legal solved): {cell_acc*100:.2f}%")
print(f"Validation full-puzzle exact match (legal solved): {puzzle_acc*100:.2f}%")

# ---- Plot loss/accuracy history ----
def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    if 'cell_acc' in history.history:
        plt.plot(history.history['cell_acc'], label='train cell_acc')
    if 'val_cell_acc' in history.history:
        plt.plot(history.history['val_cell_acc'], label='val cell_acc')
    plt.legend()
    plt.title('Training history')
    plt.xlabel('Epoch')
    plt.show()

plot_history(history)

# ---- Save model to Google Drive ----
MODEL_OUT = '/content/drive/MyDrive/Models/sudoku_model.keras'
TFLITE_OUT = '/content/drive/MyDrive/Models/sudoku_model.tflite'
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Save Keras model
model.save(MODEL_OUT)
print("Saved Keras model to", MODEL_OUT)

# ---- Recommended TFLite conversion: cast weights to float32 first ----
# This block attempts a clean conversion using native TFLite ops by ensuring weights are float32.
# If that fails, it will try a few fallbacks and finally enable SELECT_TF_OPS as last resort.

# 1) Cast weights to float32 in-place
weights = model.get_weights()
weights32 = [w.astype(np.float32) for w in weights]
model.set_weights(weights32)

# 2) Set policy to float32 to avoid creating float16 ops in conversion artifacts
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('float32')
    print('Set Keras global policy to float32 for conversion')
except Exception:
    pass

# 3) Save a SavedModel and convert from it (often more robust)
saved_model_dir = '/tmp/saved_sudoku_model'
if os.path.exists(saved_model_dir):
    import shutil
    shutil.rmtree(saved_model_dir)

tf.saved_model.save(model, saved_model_dir)

# Try conversion from saved model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# Optional: you can enable optimizations here if you want
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    print('TFLite conversion succeeded (from saved_model).')
except Exception as e:
    print('Primary conversion (from_saved_model) failed with:', e)
    print('Trying converter.from_keras_model fallback...')
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print('TFLite conversion succeeded (from_keras_model).')
    except Exception as e2:
        print('Fallback conversion failed with:', e2)
        print('Last resort: enabling SELECT_TF_OPS fallback (requires TF Select runtime at inference).')
        # Last resort: enable TF Select ops so converter keeps unsupported TF ops
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,      # builtin TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS         # allow TF ops fallback (Flex/Select)
        ]
        # Note: this increases runtime size and requires the TF Select delegate at inference.
        tflite_model = converter.convert()

# Write tflite file
with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)
print('Saved TensorFlow Lite model to', TFLITE_OUT)
print(f'TensorFlow Lite model size: {len(tflite_model) / 1024 / 1024:.2f} MB')

print('\nRecommendation: Casting weights to float32 before conversion (as done here) keeps the .tflite native and small.\nIf you plan to run on Android without the TF Select delegate, this is the right path.')