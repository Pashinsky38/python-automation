"""
BTD6 Imitation Learning Project
================================
An AI system that learns to play Bloons TD 6 through imitation learning.
The system observes human gameplay, collects screen captures and input actions,
then trains a neural network to replicate the player's behavior.

Features:
- Data collection from live gameplay
- Temporal sequence processing with LSTM
- Position-aware action prediction for precise tower placement
- Class-balanced training to handle action imbalance
- Automated gameplay using trained models

Author: AI Assistant
Date: 2024
"""

import os
import time
import json
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pynput import keyboard, mouse
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Controller as MouseController
import pyautogui
from datetime import datetime
import queue
from collections import deque, Counter

# -----------------------------
# Data collector (Enhanced)
# -----------------------------
class BTD6DataCollector:
    """Enhanced collector with position tracking and richer action representation"""

    def __init__(self, game_region=None, fps=10):
        self.game_region = game_region or {"top": 0, "left": 0, "width": 1600, "height": 900}
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # Data storage
        self.frames = []
        self.actions = []
        self.timestamps = []
        self.mouse_positions = []  # Track mouse position for each frame

        # Input tracking
        self.input_queue = queue.Queue()
        self.collecting = False
        self.current_mouse_pos = (0, 0)  # Track current mouse position

        # Input listeners
        self.keyboard_listener = None
        self.mouse_listener = None

        print("Enhanced BTD6 Data Collector initialized")
        print(f"Game region: {self.game_region}")
        print(f"Capture rate: {fps} FPS")

    def on_key_press(self, key):
        if not self.collecting:
            return
        timestamp = time.time()
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except Exception:
            key_name = str(key)
        
        action = {
            'type': 'key_press',
            'key': key_name,
            'timestamp': timestamp,
            'mouse_x': self.current_mouse_pos[0],  # Include mouse position
            'mouse_y': self.current_mouse_pos[1]
        }
        self.input_queue.put(action)

    def on_mouse_move(self, x, y):
        """Track mouse movement"""
        if not self.collecting:
            return
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            self.current_mouse_pos = (rel_x, rel_y)

    def on_mouse_click(self, x, y, button, pressed):
        if not self.collecting or not pressed:
            return
        timestamp = time.time()
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']

        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            action = {
                'type': 'mouse_click',
                'x': float(rel_x) / self.game_region['width'],  # Normalize coordinates
                'y': float(rel_y) / self.game_region['height'],
                'button': str(button),
                'timestamp': timestamp
            }
            self.input_queue.put(action)

    def on_mouse_drag(self, x, y):
        """Track mouse drag for camera movement"""
        if not self.collecting:
            return
        timestamp = time.time()
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            action = {
                'type': 'mouse_drag',
                'x': float(rel_x) / self.game_region['width'],
                'y': float(rel_y) / self.game_region['height'],
                'timestamp': timestamp
            }
            self.input_queue.put(action)

    def start_input_listeners(self):
        """Start enhanced input listeners"""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
            on_move=self.on_mouse_move
        )
        self.keyboard_listener.start()
        self.mouse_listener.start()
        print("Enhanced input listeners started")

    def stop_input_listeners(self):
        """Stop input listeners"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        print("Input listeners stopped")

    def capture_frame(self, sct):
        """Capture a single frame from the game region"""
        try:
            screenshot = sct.grab(self.game_region)
            frame = np.array(screenshot)
            # MSS returns BGRA, convert to BGR for consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame, time.time()
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None, None

    def collect_data(self, duration=300):
        """Enhanced data collection with mouse position tracking"""
        print(f"Starting enhanced data collection for {duration} seconds...")
        print("Move mouse to upper-left corner to trigger PyAutoGUI fail-safe.")
        self.collecting = True
        self.start_input_listeners()
        start_time = time.time()
        frame_count = 0

        try:
            with mss.mss() as sct:
                while self.collecting and (time.time() - start_time) < duration:
                    loop_start = time.time()
                    frame, timestamp = self.capture_frame(sct)
                    
                    if frame is not None:
                        self.frames.append(frame)
                        self.timestamps.append(timestamp)
                        self.mouse_positions.append(self.current_mouse_pos)  # Store mouse position
                        frame_count += 1
                        
                        if frame_count % 50 == 0:
                            print(f"Captured {frame_count} frames...")

                    # Drain the input queue
                    recent_actions = []
                    while True:
                        try:
                            action = self.input_queue.get_nowait()
                            recent_actions.append(action)
                        except queue.Empty:
                            break

                    self.actions.append(recent_actions)

                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.frame_interval - elapsed)
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        finally:
            self.collecting = False
            self.stop_input_listeners()
            print(f"Data collection complete. Captured {len(self.frames)} frames")

    def save_data(self, filename_prefix="btd6_data"):
        """Save enhanced collected data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_frames = len(self.frames)

        if num_frames == 0:
            raise RuntimeError("No frames collected. Nothing to save.")

        LARGE_THRESHOLD = 1000

        if num_frames > LARGE_THRESHOLD:
            folder = f"{filename_prefix}_frames_{timestamp}"
            os.makedirs(folder, exist_ok=True)
            print(f"Saving {num_frames} frames to folder {folder}...")
            for i, frame in enumerate(self.frames):
                path = os.path.join(folder, f"frame_{i:06d}.png")
                cv2.imwrite(path, frame)
                if (i + 1) % 200 == 0:
                    print(f"Written {i+1}/{num_frames} frames...")
            frames_file = folder
            print(f"Frames saved to folder: {frames_file}")
        else:
            frames_file = f"{filename_prefix}_frames_{timestamp}.npz"
            np.savez_compressed(frames_file, frames=np.array(self.frames, dtype=np.uint8))
            print(f"Frames saved to {frames_file}")

        # Save enhanced action data
        data_file = f"{filename_prefix}_actions_{timestamp}.json"
        data = {
            'actions': self.actions,
            'timestamps': self.timestamps,
            'mouse_positions': self.mouse_positions,  # Include mouse positions
            'game_region': self.game_region,
            'fps': self.fps
        }
        with open(data_file, 'w') as f:
            json.dump(data, f)
        print(f"Enhanced actions saved to {data_file}")

        return frames_file, data_file

# -----------------------------
# Enhanced Dataset with Class Balancing
# -----------------------------
class BTD6Dataset(Dataset):
    """Enhanced dataset with position information and temporal context"""

    def __init__(self, frames_source, actions_file, sequence_length=5):
        """
        frames_source: path to .npz file OR path to folder containing frame_XXXXXX.png
        actions_file: JSON file saved by collector
        sequence_length: number of frames to use as temporal context
        """
        self.frames_source = frames_source
        self.sequence_length = sequence_length

        # Load actions
        with open(actions_file, 'r') as f:
            data = json.load(f)
        self.actions = data.get('actions', [])
        self.timestamps = data.get('timestamps', [])
        self.mouse_positions = data.get('mouse_positions', [])
        self.game_region = data.get('game_region', None)
        self.fps = data.get('fps', None)

        # Load frames
        if os.path.isdir(frames_source):
            files = sorted([os.path.join(frames_source, fn) for fn in os.listdir(frames_source) if fn.endswith('.png')])
            if not files:
                raise RuntimeError("No PNG frames found in folder.")
            self.frame_paths = files
            self.frames = None
            self._len = len(self.frame_paths)
        elif frames_source.endswith('.npz'):
            arr = np.load(frames_source)
            self.frames = arr['frames']
            self.frame_paths = None
            self._len = len(self.frames)
        else:
            raise ValueError("frames_source must be a folder or .npz file")

        # Ensure consistent lengths
        minlen = min(self._len, len(self.actions), len(self.timestamps))
        if len(self.mouse_positions) > 0:
            minlen = min(minlen, len(self.mouse_positions))
        
        if minlen != self._len:
            print(f"Warning: truncating to min length: {minlen}")
            self._len = minlen
            if self.frames is not None:
                self.frames = self.frames[:minlen]
            if self.frame_paths is not None:
                self.frame_paths = self.frame_paths[:minlen]
            self.actions = self.actions[:minlen]
            self.timestamps = self.timestamps[:minlen]
            if self.mouse_positions:
                self.mouse_positions = self.mouse_positions[:minlen]

        # Process labels with enhanced action representation
        self.labels = self._process_enhanced_actions()
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()
        
        print(f"Enhanced dataset loaded: {self._len} frames")
        print(f"Class distribution: {self.class_weights}")

    def _process_enhanced_actions(self):
        """Process actions into enhanced labels including positions"""
        labels = []
        
        for i in range(self._len):
            frame_actions = self.actions[i] if i < len(self.actions) else []
            mouse_pos = self.mouse_positions[i] if i < len(self.mouse_positions) and self.mouse_positions else (0, 0)
            
            # Enhanced label: [action_type, x_coord, y_coord, needs_position]
            # action_type: 0=none, 1=left_click, 2=right_click, 3=key, 4=drag
            # needs_position: whether this action requires position (for masking loss)
            label = {'action': 0, 'x': 0.0, 'y': 0.0, 'needs_position': False}
            
            if frame_actions and isinstance(frame_actions, list):
                for action in frame_actions:
                    if not isinstance(action, dict):
                        continue
                    
                    action_type = action.get('type')
                    if action_type == 'mouse_click':
                        btn = action.get('button', '').lower()
                        label['action'] = 1 if 'left' in btn else 2
                        label['x'] = action.get('x', 0.0)
                        label['y'] = action.get('y', 0.0)
                        label['needs_position'] = True
                        break  # Prioritize mouse clicks
                    elif action_type == 'key_press':
                        label['action'] = 3
                        label['needs_position'] = False
                    elif action_type == 'mouse_drag':
                        label['action'] = 4
                        label['x'] = action.get('x', 0.0)
                        label['y'] = action.get('y', 0.0)
                        label['needs_position'] = True
            
            labels.append(label)
        
        return labels

    def _calculate_class_weights(self):
        """Calculate weights for handling class imbalance"""
        action_counts = Counter([label['action'] for label in self.labels])
        total = sum(action_counts.values())
        num_classes = 5  # 0-4 action types
        
        # Calculate inverse frequency weights
        weights = []
        for i in range(num_classes):
            count = action_counts.get(i, 1)  # Avoid division by zero
            weight = total / (num_classes * count)
            weights.append(weight)
        
        # Log class distribution
        print("Action distribution:")
        action_names = ['none', 'left_click', 'right_click', 'key', 'drag']
        for i in range(num_classes):
            count = action_counts.get(i, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {action_names[i]}: {count} ({percentage:.1f}%)")
        
        return torch.FloatTensor(weights)

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler to handle class imbalance"""
        weights = []
        for label in self.labels:
            action = label['action']
            weights.append(self.class_weights[action].item())
        return weights

    def __len__(self):
        # Account for sequence length
        return max(0, self._len - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Get sequence of frames for temporal context
        frames_seq = []
        
        for i in range(idx, idx + self.sequence_length):
            if i >= self._len:
                i = self._len - 1  # Repeat last frame if needed
            
            if self.frames is not None:
                frame = self.frames[i]
            else:
                path = self.frame_paths[i]
                frame = cv2.imread(path)
                if frame is None:
                    raise RuntimeError(f"Failed to read frame file {path}")
            
            # Frame is already in BGR from cv2, keep it consistent
            # Enhanced preprocessing - less aggressive resize to preserve details
            frame = cv2.resize(frame, (320, 180))  # 16:9 aspect ratio, more detail
            frame = frame.astype(np.float32) / 255.0
            frames_seq.append(frame)
        
        # Stack frames along channel dimension for temporal info
        frames_array = np.stack(frames_seq, axis=0)  # [seq_len, H, W, C]
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # [seq_len, C, H, W]
        
        # Get label for the last frame in sequence
        label = self.labels[min(idx + self.sequence_length - 1, self._len - 1)]
        
        # Convert to tensors
        frames_tensor = torch.tensor(frames_array, dtype=torch.float32)
        action_tensor = torch.tensor(label['action'], dtype=torch.long)
        pos_tensor = torch.tensor([label['x'], label['y']], dtype=torch.float32)
        needs_pos_tensor = torch.tensor(label['needs_position'], dtype=torch.bool)
        
        return frames_tensor, action_tensor, pos_tensor, needs_pos_tensor

# -----------------------------
# Enhanced Model with Temporal Processing
# -----------------------------
class BTD6ImitationNet(nn.Module):
    """Enhanced network with temporal processing and position regression"""

    def __init__(self, num_actions=5, sequence_length=5):
        super(BTD6ImitationNet, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN for frame feature extraction
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 7))  # Fixed size output
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=128 * 4 * 7,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Separate heads for action classification and position regression
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions)
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # x, y coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process each frame through CNN
        frame_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            features = self.frame_encoder(frame)
            features = features.view(batch_size, -1)
            frame_features.append(features)
        
        # Stack and process through LSTM
        frame_features = torch.stack(frame_features, dim=1)
        lstm_out, _ = self.lstm(frame_features)
        
        # Use last LSTM output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Get action and position predictions
        action_logits = self.action_head(last_output)
        positions = self.position_head(last_output)
        
        return action_logits, positions

# -----------------------------
# Enhanced Trainer with Masked Position Loss
# -----------------------------
class BTD6Trainer:
    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        # Use class weights for imbalanced data
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.action_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.position_criterion = nn.MSELoss(reduction='none')  # Element-wise loss for masking
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Loss weights
        self.action_weight = 1.0
        self.position_weight = 2.0  # Position is crucial in BTD6

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        action_correct = 0
        total = 0
        position_error = 0.0
        position_samples = 0

        for batch_idx, (frames, actions, positions, needs_pos) in enumerate(train_loader):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            positions = positions.to(self.device)
            needs_pos = needs_pos.to(self.device)
            
            self.optimizer.zero_grad()
            
            action_logits, pred_positions = self.model(frames)
            
            # Calculate action loss
            action_loss = self.action_criterion(action_logits, actions)
            
            # Calculate masked position loss (only for actions that need position)
            pos_loss_raw = self.position_criterion(pred_positions, positions)
            pos_loss_per_sample = pos_loss_raw.mean(dim=1)  # Average over x,y
            
            # Mask: only apply loss where position is needed
            masked_pos_loss = pos_loss_per_sample * needs_pos.float()
            
            # Average over samples that need position (avoid division by zero)
            num_pos_samples = needs_pos.sum().item()
            if num_pos_samples > 0:
                position_loss = masked_pos_loss.sum() / num_pos_samples
            else:
                position_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            loss = self.action_weight * action_loss + self.position_weight * position_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            total_loss += loss.item()
            pred = action_logits.argmax(dim=1)
            action_correct += (pred == actions).sum().item()
            total += actions.size(0)
            
            if num_pos_samples > 0:
                position_error += position_loss.item() * num_pos_samples
                position_samples += num_pos_samples

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Action Loss: {action_loss.item():.4f}, '
                      f'Position Loss: {position_loss.item():.4f}, '
                      f'Pos Samples: {num_pos_samples}')

        avg_loss = total_loss / max(1, len(train_loader))
        action_accuracy = 100.0 * action_correct / max(1, total)
        avg_position_error = position_error / max(1, position_samples) if position_samples > 0 else 0
        
        return avg_loss, action_accuracy, avg_position_error

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        action_correct = 0
        total = 0
        position_error = 0.0
        position_samples = 0
        
        with torch.no_grad():
            for frames, actions, positions, needs_pos in val_loader:
                frames = frames.to(self.device)
                actions = actions.to(self.device)
                positions = positions.to(self.device)
                needs_pos = needs_pos.to(self.device)
                
                action_logits, pred_positions = self.model(frames)
                
                # Calculate action loss
                action_loss = self.action_criterion(action_logits, actions)
                
                # Calculate masked position loss
                pos_loss_raw = self.position_criterion(pred_positions, positions)
                pos_loss_per_sample = pos_loss_raw.mean(dim=1)
                masked_pos_loss = pos_loss_per_sample * needs_pos.float()
                
                num_pos_samples = needs_pos.sum().item()
                if num_pos_samples > 0:
                    position_loss = masked_pos_loss.sum() / num_pos_samples
                else:
                    position_loss = torch.tensor(0.0, device=self.device)
                
                loss = self.action_weight * action_loss + self.position_weight * position_loss
                
                val_loss += loss.item()
                pred = action_logits.argmax(dim=1)
                action_correct += (pred == actions).sum().item()
                total += actions.size(0)
                
                if num_pos_samples > 0:
                    position_error += position_loss.item() * num_pos_samples
                    position_samples += num_pos_samples
        
        avg_loss = val_loss / max(1, len(val_loader))
        action_accuracy = 100.0 * action_correct / max(1, total)
        avg_position_error = position_error / max(1, position_samples) if position_samples > 0 else 0
        
        return avg_loss, action_accuracy, avg_position_error

    def train(self, train_loader, val_loader, epochs=50, save_path='best_btd6_model.pth'):
        print("Starting enhanced training with masked position loss...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'\n--- Epoch {epoch+1}/{epochs} ---')
            
            train_loss, train_acc, train_pos_err = self.train_epoch(train_loader)
            val_loss, val_acc, val_pos_err = self.validate(val_loader)
            
            print(f'Train - Loss: {train_loss:.4f}, Action Acc: {train_acc:.2f}%, Pos Error: {train_pos_err:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Action Acc: {val_acc:.2f}%, Pos Error: {val_pos_err:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_pos_err': val_pos_err
                }, save_path)
                print(f'✓ New best model saved! Val Loss: {val_loss:.4f}')

# -----------------------------
# Enhanced Controller with Fixed Execution
# -----------------------------
class BTD6Controller:
    def __init__(self, model_path, game_region=None, sequence_length=5, num_actions=5):
        self.game_region = game_region or {"top": 0, "left": 0, "width": 1600, "height": 900}
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = BTD6ImitationNet(num_actions=num_actions, sequence_length=sequence_length)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Frame buffer for temporal context
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Controllers
        self.kb = KeyController()
        
        # Fixed: Format checkpoint values properly before printing
        val_loss = checkpoint.get('val_loss', None)
        val_acc = checkpoint.get('val_acc', None)
        val_pos_err = checkpoint.get('val_pos_err', None)
        
        # Format the values for display
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        acc_str = f"{val_acc:.2f}%" if val_acc is not None else "N/A"
        pos_err_str = f"{val_pos_err:.4f}" if val_pos_err is not None else "N/A"
        
        print(f"Enhanced BTD6 Controller loaded")
        print(f"Model checkpoint - Val Loss: {loss_str}, Val Acc: {acc_str}, Pos Error: {pos_err_str}")
        print(f"Using device: {self.device}")

    def preprocess_frame(self, frame):
        # Frame is already BGR from mss capture, keep consistent
        frame = cv2.resize(frame, (320, 180))
        frame = frame.astype(np.float32) / 255.0
        return frame

    def execute_enhanced_action(self, action_id, x_norm, y_norm):
        """Execute action with precise positioning - Fixed with consistent pyautogui usage"""
        # Convert normalized coordinates to screen coordinates
        x = int(self.game_region['left'] + x_norm * self.game_region['width'])
        y = int(self.game_region['top'] + y_norm * self.game_region['height'])
        
        # Log predicted positions for debugging
        action_names = ['none', 'left_click', 'right_click', 'key', 'drag']
        
        if action_id == 0:
            # No action
            return
        elif action_id == 1:
            # Left click at specific position
            pyautogui.click(x, y)
            print(f"[{action_names[action_id]}] at ({x}, {y}) - norm: ({x_norm:.3f}, {y_norm:.3f})")
        elif action_id == 2:
            # Right click at specific position
            pyautogui.rightClick(x, y)
            print(f"[{action_names[action_id]}] at ({x}, {y}) - norm: ({x_norm:.3f}, {y_norm:.3f})")
        elif action_id == 3:
            # Key press (space for start/pause)
            self.kb.press(Key.space)
            self.kb.release(Key.space)
            print(f"[{action_names[action_id]}] Space key")
        elif action_id == 4:
            # Fixed: Use pyautogui consistently for drag
            current_x, current_y = pyautogui.position()
            pyautogui.dragTo(x, y, duration=0.2, button='left')
            print(f"[{action_names[action_id]}] from ({current_x}, {current_y}) to ({x}, {y})")

    def play(self, duration=60, fps=5):
        print(f"Starting enhanced automated play for {duration} seconds...")
        start_time = time.time()
        frame_count = 0
        frame_interval = 1.0 / fps
        
        # Action statistics tracking
        action_counts = Counter()

        try:
            with mss.mss() as sct:
                # Initialize frame buffer with first frames
                print("Initializing frame buffer...")
                for _ in range(self.sequence_length):
                    shot = sct.grab(self.game_region)
                    frame = np.array(shot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Consistent BGR
                    processed = self.preprocess_frame(frame)
                    self.frame_buffer.append(processed)
                    time.sleep(0.1)
                
                print("Starting gameplay...")
                while (time.time() - start_time) < duration:
                    loop_start = time.time()
                    
                    # Capture new frame
                    shot = sct.grab(self.game_region)
                    frame = np.array(shot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Consistent BGR
                    processed = self.preprocess_frame(frame)
                    self.frame_buffer.append(processed)
                    
                    # Prepare input tensor
                    frames_array = np.stack(list(self.frame_buffer), axis=0)
                    frames_array = np.transpose(frames_array, (0, 3, 1, 2))
                    input_tensor = torch.tensor(frames_array, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get model prediction
                    with torch.no_grad():
                        action_logits, positions = self.model(input_tensor)
                        action = int(action_logits.argmax().item())
                        x_norm = positions[0, 0].item()
                        y_norm = positions[0, 1].item()
                    
                    # Execute action and track statistics
                    self.execute_enhanced_action(action, x_norm, y_norm)
                    action_counts[action] += 1
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"Processed {frame_count} frames...")
                    
                    # Maintain FPS
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_interval - elapsed)
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nAutomated play stopped by user")
        
        # Print action statistics
        print(f"\nAutomated play complete. Processed {frame_count} frames")
        print("\nAction statistics:")
        action_names = ['none', 'left_click', 'right_click', 'key', 'drag']
        for action_id in range(5):
            count = action_counts[action_id]
            percentage = (count / frame_count * 100) if frame_count > 0 else 0
            print(f"  {action_names[action_id]}: {count} ({percentage:.1f}%)")

# -----------------------------
# Main CLI (Enhanced with Class Balancing)
# -----------------------------
def main():
    pyautogui.PAUSE = 0.01
    pyautogui.FAILSAFE = True

    print("=" * 50)
    print("BTD6 AI Imitation Learning System (Fixed)")
    print("=" * 50)
    print("Fixes applied:")
    print("✓ Checkpoint loading with safe formatting")
    print("✓ Masked position loss (only for clicks/drags)")
    print("✓ Consistent pyautogui usage for actions")
    print("✓ Class weighting for imbalanced data")
    print("✓ Position logging for debugging")
    print("=" * 50)

    mode = input("\nChoose mode:\n1. Collect data\n2. Train model\n3. Play with trained model\nEnter choice (1/2/3): ").strip()

    if mode == '1':
        print("\n--- Data Collection Mode ---")
        fps = int(input("FPS (default 5, recommended 5-10): ") or "5")
        print("\nPreparing to collect data...")
        print("1. Start BTD6 and enter a game")
        print("2. Position the game window")
        print("3. Press Enter when ready")
        input("\nPress Enter to start collection...")
        
        collector = BTD6DataCollector(fps=fps)
        duration = int(input("Enter collection duration in seconds (default 300): ") or "300")
        
        print("\nCollection starting in 3 seconds...")
        time.sleep(3)
        
        collector.collect_data(duration)
        frames_file, actions_file = collector.save_data()
        
        print("\n✓ Data collection complete!")
        print(f"Frames: {frames_file}")
        print(f"Actions: {actions_file}")

    elif mode == '2':
        print("\n--- Training Mode ---")
        frames_source = input("Enter frames source (.npz or folder): ").strip()
        actions_file = input("Enter actions file path (.json): ").strip()
        
        if not os.path.exists(frames_source):
            print("❌ Frames source not found!")
            return
        if not os.path.exists(actions_file):
            print("❌ Actions file not found!")
            return

        print("\nLoading dataset...")
        sequence_length = int(input("Sequence length for temporal context (default 5): ") or "5")
        dataset = BTD6Dataset(frames_source, actions_file, sequence_length=sequence_length)
        
        # Get class weights and sample weights for balancing
        class_weights = dataset.class_weights
        sample_weights = dataset.get_sample_weights()
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create weighted sampler for balanced training
        train_indices = train_dataset.indices
        train_weights = [sample_weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        
        # Create data loaders
        batch_size = int(input("Batch size (default 8): ") or "8")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = BTD6ImitationNet(num_actions=5, sequence_length=sequence_length)
        trainer = BTD6Trainer(model, device, class_weights=class_weights)
        
        epochs = int(input("Number of epochs (default 30): ") or "30")
        trainer.train(train_loader, val_loader, epochs, save_path='best_btd6_model_fixed.pth')
        
        print("\n✓ Training complete!")

    elif mode == '3':
        print("\n--- Play Mode ---")
        model_path = input("Enter model path (default 'best_btd6_model_fixed.pth'): ") or "best_btd6_model_fixed.pth"
        
        if not os.path.exists(model_path):
            print("❌ Model not found!")
            return
        
        sequence_length = int(input("Sequence length (must match training, default 5): ") or "5")
        controller = BTD6Controller(model_path, sequence_length=sequence_length)
        
        duration = int(input("Play duration in seconds (default 60): ") or "60")
        
        print("\n1. Start BTD6 and enter a game")
        print("2. Position the game window")
        print("3. Press Enter when ready")
        input("\nPress Enter to start automated play...")
        
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        controller.play(duration, fps=5)
        print("\n✓ Automated play complete!")

    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()