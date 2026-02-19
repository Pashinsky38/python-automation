#!/usr/bin/env python3
"""
Wordle Solver with Enhanced Tkinter GUI  (Optimised)

Key changes vs original:
  - Integer feedback encoding (base-3, 0-242) replaces tuple/string keys.
  - numpy bincount-based entropy: ~10-50x faster than Python defaultdict loop.
  - _heuristic_score now caches the letter-frequency table per candidate set
    (original recomputed it from scratch on *every* call – O(n_candidates) waste).
  - get_feedback tightened: plain int array instead of Counter; lru_cache kept.
  - allowed_guesses_set cached on WordleSolver for O(1) membership tests.
  - suggest_next pool-building deduplication done with a set, not repeated list ops.

Usage:
  python wordle_solver_gui.py --words valid-wordle-words.txt

Dependencies: Python 3.x, tkinter, numpy
"""

from collections import defaultdict
import math
import argparse
import sys
import os
from functools import lru_cache
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# ---------- constants & tuning ----------
DEFAULT_WORD_FILE = "valid-wordle-words.txt"
BEST_STARTING_WORDS = ["salet", "roate", "raise", "crane", "slate", "crate", "trace", "carte"]
GET_FEEDBACK_CACHE = 200_000
HEURISTIC_TOP_K = 400
CANDIDATE_BONUS = 1e-3

# Feedback integer constants
WIN_INT = 242   # (2,2,2,2,2) in base-3

# Color scheme
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'accent': '#0f3460',
    'highlight': '#533483',
    'text': '#eee',
    'gray': '#787c7e',
    'yellow': '#c9b458',
    'green': '#6aaa64',
    'button': '#818384',
    'button_hover': '#9a9b9c',
    'success': '#538d4e',
    'error': '#d94744'
}

# ---------- feedback utilities ----------

@lru_cache(maxsize=GET_FEEDBACK_CACHE)
def get_feedback(guess: str, solution: str) -> tuple:
    """Return feedback as a (int,int,int,int,int) tuple (0=gray,1=yellow,2=green).
    Kept for filter_candidates compatibility.
    """
    feedback = [0] * 5
    # Use a fixed-size int array instead of Counter – avoids hash overhead
    remaining = [0] * 26
    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 2
        else:
            remaining[ord(solution[i]) - 97] += 1
    for i in range(5):
        if feedback[i] == 2:
            continue
        idx = ord(guess[i]) - 97
        if remaining[idx] > 0:
            feedback[i] = 1
            remaining[idx] -= 1
    return tuple(feedback)


@lru_cache(maxsize=GET_FEEDBACK_CACHE)
def get_feedback_int(guess: str, solution: str) -> int:
    """Return feedback encoded as a base-3 integer (0-242).
    Used internally for fast numpy operations.
    """
    v = 0
    remaining = [0] * 26
    fb = [0] * 5
    for i in range(5):
        if guess[i] == solution[i]:
            fb[i] = 2
        else:
            remaining[ord(solution[i]) - 97] += 1
    for i in range(5):
        if fb[i] == 2:
            continue
        idx = ord(guess[i]) - 97
        if remaining[idx] > 0:
            fb[i] = 1
            remaining[idx] -= 1
    # base-3 encode: position 0 is most significant
    for x in fb:
        v = v * 3 + x
    return v


def feedback_tuple_to_int(fb: tuple) -> int:
    v = 0
    for x in fb:
        v = v * 3 + x
    return v


def feedback_int_to_tuple(n: int) -> tuple:
    result = [0] * 5
    for i in range(4, -1, -1):
        result[i] = n % 3
        n //= 3
    return tuple(result)


# ---------- solver core ----------
class WordleSolver:
    def __init__(self, candidates):
        self.candidates = list(dict.fromkeys(w.lower() for w in candidates if len(w) == 5 and w.isalpha()))
        self.candidates_set = set(self.candidates)
        self.history = []
        self.initial_size = len(self.candidates)
        self._entropy_cache: dict = {}
        # Cached letter-frequency table (invalidated when candidates change)
        self._letter_freq: np.ndarray | None = None   # shape (26,)
        self._allowed_guesses_set: set | None = None

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def filter_candidates(self, guess: str, feedback: tuple):
        fb_int = feedback_tuple_to_int(feedback)
        new_candidates = [w for w in self.candidates if get_feedback_int(guess, w) == fb_int]
        removed = len(self.candidates) - len(new_candidates)
        self.candidates = new_candidates
        self.candidates_set = set(new_candidates)
        self._entropy_cache.clear()
        self._letter_freq = None   # invalidate cached freq table
        return removed

    # ------------------------------------------------------------------
    # Entropy (numpy-accelerated)
    # ------------------------------------------------------------------

    def _compute_entropy(self, guess: str, candidates: list):
        cached = self._entropy_cache.get(guess)
        if cached is not None:
            return cached

        # Build uint8 array of feedback integers – one entry per candidate
        n = len(candidates)
        fb_arr = np.empty(n, dtype=np.uint8)
        g_int = get_feedback_int  # local alias saves attribute lookup in loop
        for j, sol in enumerate(candidates):
            fb_arr[j] = g_int(guess, sol)

        # bincount over 0-242 → partition counts
        counts = np.bincount(fb_arr, minlength=243)
        nonzero = counts[counts > 0].astype(np.float64)
        p = nonzero / n
        entropy = float(-np.dot(p, np.log2(p)))
        result = (entropy, int(nonzero.size))
        self._entropy_cache[guess] = result
        return result

    # ------------------------------------------------------------------
    # Heuristic scoring (fixed: freq computed once, not per-guess)
    # ------------------------------------------------------------------

    def _get_letter_freq(self) -> np.ndarray:
        """Return a (26,) array counting unique letter occurrences across candidates."""
        if self._letter_freq is not None:
            return self._letter_freq
        freq = np.zeros(26, dtype=np.int32)
        for w in self.candidates:
            for ch in set(w):
                freq[ord(ch) - 97] += 1
        self._letter_freq = freq
        return freq

    def _heuristic_score(self, guess: str) -> int:
        """Sum of letter-frequency scores for unique letters in guess."""
        freq = self._get_letter_freq()
        return int(sum(freq[ord(ch) - 97] for ch in set(guess)))

    # ------------------------------------------------------------------
    # Starting words
    # ------------------------------------------------------------------

    def _get_best_starting_words(self, allowed_set: set, top_n: int = 5):
        available = [w for w in BEST_STARTING_WORDS if w in allowed_set]
        if len(available) < top_n:
            extras = ["adieu", "audio", "ouija", "arose", "irate", "stare", "tears", "store"]
            for w in extras:
                if w in allowed_set and w not in available:
                    available.append(w)
                if len(available) >= top_n:
                    break
        return [(w, 5.0) for w in available[:top_n]]

    # ------------------------------------------------------------------
    # Main suggestion engine
    # ------------------------------------------------------------------

    def suggest_next(self, allowed_guesses=None, top_n: int = 1):
        if allowed_guesses is None:
            allowed_guesses = self.candidates

        if not self.candidates:
            return []
        if len(self.candidates) == 1:
            return [(self.candidates[0], float('inf'))]
        if len(self.candidates) <= 2:
            return [(w, 1.0) for w in self.candidates[:top_n]]

        # Cache set for O(1) membership tests
        if self._allowed_guesses_set is None or len(self._allowed_guesses_set) != len(allowed_guesses):
            self._allowed_guesses_set = set(allowed_guesses)
        allowed_set = self._allowed_guesses_set

        if len(self.candidates) == self.initial_size and not self.history:
            return self._get_best_starting_words(allowed_set, top_n)

        pool = allowed_guesses

        # Pool trimming (same logic as before, but set-based for O(1) lookup)
        if len(pool) > 2000 and len(self.candidates) > 50:
            priority_pool = [w for w in pool if w in self.candidates_set]
            other_pool = [w for w in pool if w not in self.candidates_set]
            pool = priority_pool + other_pool[:800]
        elif len(pool) > 5000:
            pool = [w for w in pool if w in self.candidates_set][:1500]

        # Heuristic pre-filter: score all words in pool, keep top-k
        compute_pool = pool
        if len(pool) > HEURISTIC_TOP_K:
            # _heuristic_score now uses the cached freq table – no redundant recompute
            scored = sorted(pool, key=self._heuristic_score, reverse=True)
            compute_pool = scored[:HEURISTIC_TOP_K]

        # Entropy computation (numpy-accelerated bincount)
        results = []
        cands = self.candidates  # local ref
        is_candidate = self.candidates_set
        for guess in compute_pool:
            entropy, part_count = self._compute_entropy(guess, cands)
            bonus = CANDIDATE_BONUS if guess in is_candidate else 0.0
            results.append((guess, entropy + bonus, part_count))

        # Ensure every candidate appears (in case heuristic filtered them out)
        if compute_pool is not pool:
            seen = set(compute_pool)
            for w in self.candidates:
                if w not in seen:
                    entropy, part_count = self._compute_entropy(w, cands)
                    results.append((w, entropy + CANDIDATE_BONUS, part_count))
                    if len(results) >= HEURISTIC_TOP_K + 50:
                        break

        results.sort(key=lambda x: (-x[1], -x[2]))
        return [(w, s) for w, s, _ in results[:top_n]]

    def add_history(self, guess: str, feedback: tuple):
        self.history.append((guess, feedback))
        removed = self.filter_candidates(guess, feedback)
        return removed


# ---------- I/O helpers ----------

def load_words_from_file(path):
    words = []
    seen = set()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == 5 and w.isalpha() and w not in seen:
                words.append(w)
                seen.add(w)
    return words


# ---------- Custom Widgets ----------

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, width=120, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['panel'],
                         highlightthickness=0, **kwargs)
        self.command = command
        self.width = width
        self.height = height
        self.rect = self.create_rectangle(2, 2, width - 2, height - 2,
                                          fill=COLORS['accent'], outline='')
        self.text_id = self.create_text(width // 2, height // 2, text=text,
                                        fill=COLORS['text'], font=('Arial', 11, 'bold'))
        self.bind('<Button-1>', lambda e: self.on_click())
        self.bind('<Enter>', lambda e: self.on_hover())
        self.bind('<Leave>', lambda e: self.on_leave())

    def on_hover(self): self.itemconfig(self.rect, fill=COLORS['highlight'])
    def on_leave(self): self.itemconfig(self.rect, fill=COLORS['accent'])
    def on_click(self):
        if self.command:
            self.command()


class FeedbackBox(tk.Canvas):
    def __init__(self, parent, index, callback, **kwargs):
        super().__init__(parent, width=60, height=60, bg=COLORS['panel'],
                         highlightthickness=0, **kwargs)
        self.index = index
        self.callback = callback
        self.state = 0
        self.rect = self.create_rectangle(5, 5, 55, 55, fill=COLORS['gray'],
                                          outline='', width=2)
        self.text_id = self.create_text(30, 30, text='', fill=COLORS['text'],
                                        font=('Arial', 20, 'bold'))
        self.bind('<Button-1>', lambda e: self.toggle())
        self.bind('<Enter>', lambda e: self.on_hover())
        self.bind('<Leave>', lambda e: self.on_leave())

    def toggle(self):
        self.state = (self.state + 1) % 3
        self.update_display()
        if self.callback:
            self.callback(self.index, self.state)

    def set_state(self, state):
        self.state = state
        self.update_display()

    def set_letter(self, letter):
        self.itemconfig(self.text_id, text=letter.upper())

    def update_display(self):
        self.itemconfig(self.rect, fill=[COLORS['gray'], COLORS['yellow'], COLORS['green']][self.state])

    def on_hover(self): self.itemconfig(self.rect, outline='white', width=2)
    def on_leave(self): self.itemconfig(self.rect, outline='', width=2)


# ---------- GUI ----------
class WordleGUI:
    def __init__(self, root, words=None):
        self.root = root
        self.root.title("Wordle Solver")
        self.root.configure(bg=COLORS['bg'])
        self.words = words or []
        self.solver = None
        self.allowed_guesses = None
        self._build_ui()
        if self.words:
            self._init_solver(self.words)

    def _build_ui(self):
        main_container = tk.Frame(self.root, bg=COLORS['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header = tk.Frame(main_container, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 20))
        tk.Label(header, text="🔤 WORDLE SOLVER", font=('Arial', 24, 'bold'),
                 fg=COLORS['text'], bg=COLORS['bg']).pack(side=tk.LEFT)
        file_frame = tk.Frame(header, bg=COLORS['bg'])
        file_frame.pack(side=tk.RIGHT)
        ModernButton(file_frame, "📁 Load Words", command=self._on_load_file,
                     width=120, height=35).pack(side=tk.LEFT, padx=5)
        self.lbl_wordfile = tk.Label(file_frame, text="No file loaded",
                                     font=('Arial', 10), fg=COLORS['button'], bg=COLORS['bg'])
        self.lbl_wordfile.pack(side=tk.LEFT, padx=10)

        # Content
        content = tk.Frame(main_container, bg=COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left panel
        left_panel = tk.Frame(content, bg=COLORS['panel'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        input_section = tk.Frame(left_panel, bg=COLORS['panel'])
        input_section.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(input_section, text="Enter Your Guess", font=('Arial', 14, 'bold'),
                 fg=COLORS['text'], bg=COLORS['panel']).pack(anchor=tk.W, pady=(0, 10))

        self.entry_guess = tk.Entry(input_section, font=('Arial', 16),
                                    bg=COLORS['accent'], fg=COLORS['text'],
                                    insertbackground=COLORS['text'], relief=tk.FLAT,
                                    justify=tk.CENTER)
        self.entry_guess.pack(fill=tk.X, ipady=8)
        self.entry_guess.bind('<KeyRelease>', self._on_guess_change)

        tk.Label(input_section, text="Click boxes to set feedback",
                 font=('Arial', 12), fg=COLORS['button'], bg=COLORS['panel']).pack(pady=(15, 5))

        fb_container = tk.Frame(input_section, bg=COLORS['panel'])
        fb_container.pack(pady=10)
        self.fb_boxes = []
        for i in range(5):
            box = FeedbackBox(fb_container, i, self._on_feedback_change)
            box.pack(side=tk.LEFT, padx=3)
            self.fb_boxes.append(box)

        btn_frame = tk.Frame(input_section, bg=COLORS['panel'])
        btn_frame.pack(pady=(20, 0))
        ModernButton(btn_frame, "✓ Apply Feedback", command=self._on_apply_feedback,
                     width=180, height=45).pack(side=tk.LEFT, padx=5)
        ModernButton(btn_frame, "↻ Reset", command=self._on_reset,
                     width=100, height=45).pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(input_section, text="", font=('Arial', 11),
                                   fg=COLORS['success'], bg=COLORS['panel'])
        self.lbl_status.pack(pady=(15, 0))

        # History
        history_section = tk.Frame(left_panel, bg=COLORS['panel'])
        history_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        tk.Label(history_section, text="History", font=('Arial', 14, 'bold'),
                 fg=COLORS['text'], bg=COLORS['panel']).pack(anchor=tk.W, pady=(0, 10))
        history_scroll_frame = tk.Frame(history_section, bg=COLORS['accent'])
        history_scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.history_canvas = tk.Canvas(history_scroll_frame, bg=COLORS['accent'],
                                        highlightthickness=0)
        self.history_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel
        right_panel = tk.Frame(content, bg=COLORS['panel'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        count_frame = tk.Frame(right_panel, bg=COLORS['panel'])
        count_frame.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(count_frame, text="Remaining Candidates", font=('Arial', 12),
                 fg=COLORS['button'], bg=COLORS['panel']).pack(anchor=tk.W)
        self.lbl_count = tk.Label(count_frame, text="0", font=('Arial', 32, 'bold'),
                                  fg=COLORS['text'], bg=COLORS['panel'])
        self.lbl_count.pack(anchor=tk.W)

        suggest_section = tk.Frame(right_panel, bg=COLORS['panel'])
        suggest_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        suggest_header = tk.Frame(suggest_section, bg=COLORS['panel'])
        suggest_header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(suggest_header, text="Top Suggestions", font=('Arial', 14, 'bold'),
                 fg=COLORS['text'], bg=COLORS['panel']).pack(side=tk.LEFT)
        tk.Label(suggest_header, text="Show:", font=('Arial', 10),
                 fg=COLORS['button'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=(20, 5))
        self.spin_suggest = tk.Spinbox(suggest_header, from_=1, to=20, width=3,
                                       font=('Arial', 10), bg=COLORS['accent'],
                                       fg=COLORS['text'], buttonbackground=COLORS['accent'],
                                       relief=tk.FLAT)
        self.spin_suggest.delete(0, tk.END)
        self.spin_suggest.insert(0, "8")
        self.spin_suggest.pack(side=tk.LEFT)
        ModernButton(suggest_header, "↻", command=self.update_ui,
                     width=40, height=30).pack(side=tk.LEFT, padx=(5, 0))

        suggest_scroll_frame = tk.Frame(suggest_section, bg=COLORS['accent'])
        suggest_scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.suggest_canvas = tk.Canvas(suggest_scroll_frame, bg=COLORS['accent'],
                                        highlightthickness=0)
        self.suggest_canvas.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_guess_change(self, event):
        guess = self.entry_guess.get().lower()
        for i, box in enumerate(self.fb_boxes):
            box.set_letter(guess[i] if i < len(guess) else '')

    def _on_feedback_change(self, index, state):
        pass

    def _on_load_file(self):
        path = filedialog.askopenfilename(
            title="Select wordlist file",
            filetypes=[("Text files", "*.txt"), ("All files", "*")])
        if not path:
            return
        try:
            words = load_words_from_file(path)
            if not words:
                messagebox.showerror("Error", f"No valid 5-letter words found in {path}")
                return
            self.lbl_wordfile.config(text=f"✓ {os.path.basename(path)}", fg=COLORS['success'])
            self.words = words
            self._init_solver(words)
            self.update_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load words: {e}")

    def _init_solver(self, words):
        self.allowed_guesses = words
        self.solver = WordleSolver(candidates=words)
        self.entry_guess.delete(0, tk.END)
        for box in self.fb_boxes:
            box.set_state(0)
            box.set_letter('')
        self.update_ui()

    def update_ui(self):
        if not self.solver:
            return
        cnt = len(self.solver.candidates)
        self.lbl_count.config(text=str(cnt))
        try:
            topn = int(self.spin_suggest.get())
        except Exception:
            topn = 8

        top = self.solver.suggest_next(allowed_guesses=self.allowed_guesses, top_n=topn)

        # Suggestions
        self.suggest_canvas.delete('all')
        y_pos = 10
        for i, (word, score) in enumerate(top):
            box_h = 50
            tag = f'sug_{i}'
            rect = self.suggest_canvas.create_rectangle(
                10, y_pos, 380, y_pos + box_h, fill=COLORS['bg'], outline='', tags=tag)
            self.suggest_canvas.create_text(
                20, y_pos + 25, text=word.upper(),
                font=('Arial', 16, 'bold'), fill=COLORS['text'], anchor=tk.W, tags=tag)
            self.suggest_canvas.create_text(
                370, y_pos + 25, text=f'{score:.3f}',
                font=('Arial', 11), fill=COLORS['button'], anchor=tk.E, tags=tag)
            self.suggest_canvas.tag_bind(tag, '<Button-1>', lambda e, w=word: self._use_word(w))
            self.suggest_canvas.tag_bind(tag, '<Enter>',
                                         lambda e, r=rect: self.suggest_canvas.itemconfig(r, fill=COLORS['accent']))
            self.suggest_canvas.tag_bind(tag, '<Leave>',
                                         lambda e, r=rect: self.suggest_canvas.itemconfig(r, fill=COLORS['bg']))
            y_pos += box_h + 5

        # History
        self.history_canvas.delete('all')
        y_pos = 10
        tile_colors = [COLORS['gray'], COLORS['yellow'], COLORS['green']]
        for guess, fb in reversed(self.solver.history[-10:]):
            for i, (letter, state) in enumerate(zip(guess, fb)):
                x = 10 + i * 55
                self.history_canvas.create_rectangle(x, y_pos, x + 50, y_pos + 50,
                                                     fill=tile_colors[state], outline='')
                self.history_canvas.create_text(x + 25, y_pos + 25, text=letter.upper(),
                                                font=('Arial', 18, 'bold'), fill='white')
            y_pos += 60

    def _use_word(self, word):
        self.entry_guess.delete(0, tk.END)
        self.entry_guess.insert(0, word)
        for i, letter in enumerate(word):
            self.fb_boxes[i].set_letter(letter)

    def _on_apply_feedback(self):
        guess = self.entry_guess.get().strip().lower()
        if not guess or len(guess) != 5 or not guess.isalpha():
            self.lbl_status.config(text="⚠ Guess must be a 5-letter word", fg=COLORS['error'])
            return

        fb = tuple(box.state for box in self.fb_boxes)
        removed = self.solver.add_history(guess, fb)
        remaining = len(self.solver.candidates)
        self.lbl_status.config(
            text=f"✓ Pruned {removed} words, {remaining} remain", fg=COLORS['success'])

        self.entry_guess.delete(0, tk.END)
        for box in self.fb_boxes:
            box.set_state(0)
            box.set_letter('')

        self.update_ui()

        if remaining == 1:
            self.lbl_status.config(
                text=f"🎉 Solved! The word is: {self.solver.candidates[0].upper()}",
                fg=COLORS['success'])

    def _on_reset(self):
        if not self.words:
            return
        if messagebox.askyesno("Reset", "Reset solver to initial state?"):
            self._init_solver(self.words)
            self.lbl_status.config(text="")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Wordle solver GUI")
    parser.add_argument("--words", help="Path to word list file", default=None)
    args = parser.parse_args()

    word_file = None
    if args.words:
        if not os.path.exists(args.words):
            print(f"Word file '{args.words}' not found.")
            sys.exit(1)
        word_file = args.words
    elif os.path.exists(DEFAULT_WORD_FILE):
        word_file = DEFAULT_WORD_FILE

    words = []
    if word_file:
        try:
            words = load_words_from_file(word_file)
            print(f"Loaded {len(words)} words from {word_file}")
        except Exception as e:
            print(f"Failed to load words: {e}")

    root = tk.Tk()
    WordleGUI(root, words=words)
    root.geometry('1000x700')
    root.minsize(900, 650)
    root.mainloop()


if __name__ == "__main__":
    main()