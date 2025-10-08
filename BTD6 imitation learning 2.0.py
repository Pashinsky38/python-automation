#!/usr/bin/env python3
"""
BTD6 Single-File AI Tool (updated)
- Adds Windows window-title detection for capture region.
Author: AI Assistant
Date: 2025
"""

import os
import sys
import time
import json
import re
import queue
import threading
import logging
from dataclasses import dataclass, asdict
from collections import Counter, deque
from datetime import datetime
from typing import List, Optional

import numpy as np
import cv2
import mss
import pyautogui
from pynput import keyboard, mouse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Try optional window detection libs
HAS_PYGETWINDOW = False
HAS_WIN32 = False
try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except Exception:
    HAS_PYGETWINDOW = False

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('btd6_single')

# -----------------------------
# Utilities: center region helper and window detection
# -----------------------------
def center_region_if_needed(region: dict) -> dict:
    """
    If left/top are 0 (defaults), compute left/top so capture region is centered
    on the primary screen.
    """
    r = dict(region)
    try:
        width = int(r.get('width', 1600))
        height = int(r.get('height', 900))
        left = int(r.get('left', 0))
        top = int(r.get('top', 0))
    except Exception:
        return r

    if left == 0 and top == 0:
        try:
            screen_w, screen_h = pyautogui.size()
            new_left = max(0, (screen_w - width) // 2)
            new_top = max(0, (screen_h - height) // 2)
            r['left'] = int(new_left)
            r['top'] = int(new_top)
            log.info(f'Auto-centered capture region to left={r["left"]}, top={r["top"]} (screen {screen_w}x{screen_h})')
        except Exception as e:
            log.warning(f'Failed to auto-center region: {e}')
    return r

def find_window_region_by_title(substring: str) -> Optional[dict]:
    """
    Try to find a visible window whose title contains the substring (case-insensitive).
    Returns a region dict {'top','left','width','height'} or None.
    Uses pygetwindow if available, else win32gui if available.
    """
    s = substring.lower().strip()
    if not s:
        return None

    # Try pygetwindow first
    if HAS_PYGETWINDOW:
        try:
            wins = gw.getWindowsWithTitle(substring)
            if not wins:
                # fallback: iterate all windows and check title contains
                all_w = gw.getAllTitles()
                matches = [t for t in all_w if s in (t or '').lower()]
                if matches:
                    wins = [gw.getWindowsWithTitle(matches[0])[0]]
            if wins:
                # pick the first visible window
                for w in wins:
                    if w.width > 0 and w.height > 0:
                        try:
                            left, top, right, bottom = w.left, w.top, w.right, w.bottom
                            width = right - left
                            height = bottom - top
                            log.info(f'pygetwindow matched "{w.title}" -> left={left}, top={top}, w={width}, h={height}')
                            return {'top': int(top), 'left': int(left), 'width': int(width), 'height': int(height)}
                        except Exception:
                            continue
        except Exception as e:
            log.warning(f'pygetwindow search failed: {e}')

    # Try win32 API
    if HAS_WIN32:
        matches = []

        def enum_cb(hwnd, _):
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd) or ''
                if s in title.lower():
                    # get rect
                    rect = win32gui.GetWindowRect(hwnd)  # (left, top, right, bottom)
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top
                    matches.append((title, left, top, width, height))
            except Exception:
                pass

        try:
            win32gui.EnumWindows(enum_cb, None)
            if matches:
                # choose largest match (likely the game window)
                matches.sort(key=lambda x: (x[3] * x[4]), reverse=True)
                title, left, top, width, height = matches[0]
                log.info(f'win32 matched "{title}" -> left={left}, top={top}, w={width}, h={height}')
                return {'top': int(top), 'left': int(left), 'width': int(width), 'height': int(height)}
        except Exception as e:
            log.warning(f'win32 enum failed: {e}')

    # Nothing found
    return None

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class RawEvent:
    timestamp: float
    event_type: str
    data: dict

# -----------------------------
# Data collector
# -----------------------------
class BTD6DataCollector:
    def __init__(self, game_region=None, fps=5, use_video=False):
        # game_region should be a dict {top,left,width,height}
        self.game_region = game_region or {"top": 0, "left": 0, "width": 1600, "height": 900}
        self.fps = max(1, int(fps))
        self.frame_interval = 1.0 / self.fps
        self.use_video = bool(use_video)

        self.event_queue = queue.Queue()
        self.raw_events: List[RawEvent] = []

        self.frame_timestamps: List[float] = []
        self.frame_count = 0
        self.frames_folder: Optional[str] = None
        self.video_writer = None
        self.current_session_id: Optional[str] = None

        self.mouse_lock = threading.Lock()
        self.current_mouse_pos = (0, 0)

        self.collecting = False
        self.keyboard_listener = None
        self.mouse_listener = None

        log.info('DataCollector initialized')

    # Input callbacks
    def on_key_press(self, key):
        if not self.collecting:
            return
        ts = time.time()
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except Exception:
            key_name = str(key)
        with self.mouse_lock:
            mx, my = self.current_mouse_pos
        evt = RawEvent(timestamp=ts, event_type='key_press', data={'key': key_name, 'mouse_x': mx, 'mouse_y': my})
        self.event_queue.put(evt)

    def on_mouse_move(self, x, y):
        if not self.collecting:
            return
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            with self.mouse_lock:
                self.current_mouse_pos = (rel_x, rel_y)

    def on_mouse_click(self, x, y, button, pressed):
        if not self.collecting or not pressed:
            return
        ts = time.time()
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            evt = RawEvent(timestamp=ts, event_type='mouse_click', data={'x': float(rel_x) / self.game_region['width'], 'y': float(rel_y) / self.game_region['height'], 'button': str(button), 'abs_x': float(rel_x), 'abs_y': float(rel_y)})
            self.event_queue.put(evt)

    def start_listeners(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click, on_move=self.on_mouse_move)
        self.keyboard_listener.start()
        self.mouse_listener.start()
        log.info('Input listeners started')

    def stop_listeners(self):
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
            self.keyboard_listener = None
        if self.mouse_listener:
            try:
                self.mouse_listener.stop()
            except Exception:
                pass
            self.mouse_listener = None
        log.info('Input listeners stopped')

    def init_frame_storage(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_session_id = ts
        if self.use_video:
            video_file = f'btd6_{ts}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_file, fourcc, self.fps, (self.game_region['width'], self.game_region['height']))
            if not self.video_writer or not self.video_writer.isOpened():
                raise RuntimeError(f'Failed to open video writer: {video_file}')
            log.info(f'Video writer: {video_file}')
            return video_file
        else:
            folder = f'btd6_frames_{ts}'
            os.makedirs(folder, exist_ok=True)
            self.frames_folder = folder
            log.info(f'Frames folder: {folder}')
            return folder

    def save_frame(self, frame, idx):
        if self.use_video and self.video_writer is not None:
            self.video_writer.write(frame)
        elif self.frames_folder:
            path = os.path.join(self.frames_folder, f'frame_{idx:06d}.png')
            cv2.imwrite(path, frame)

    def capture_frame(self, sct, idx):
        try:
            ts = time.time()
            shot = sct.grab(self.game_region)
            arr = np.array(shot)
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif arr.shape[2] == 3:
                pass
            else:
                arr = arr[:, :, :3]
            self.save_frame(arr, idx)
            return ts
        except Exception as e:
            log.warning(f'Frame capture error: {e}')
            return None

    def drain_queue(self):
        while not self.event_queue.empty():
            try:
                e = self.event_queue.get_nowait()
                self.raw_events.append(e)
            except queue.Empty:
                break

    def collect(self, duration=30):
        loc = self.init_frame_storage()
        self.collecting = True
        self.frame_count = 0
        self.frame_timestamps = []
        self.raw_events = []
        self.start_listeners()
        start = time.time()
        log.info('Starting collection... Move mouse to top-left to trigger pyautogui failsafe if needed')
        try:
            with mss.mss() as sct:
                while self.collecting and (time.time() - start) < duration:
                    loop_s = time.time()
                    ts = self.capture_frame(sct, self.frame_count)
                    if ts is not None:
                        self.frame_timestamps.append(ts)
                        self.frame_count += 1
                        self.drain_queue()
                        if self.frame_count % 50 == 0:
                            log.info(f'Captured {self.frame_count} frames, events {len(self.raw_events)}')
                    elapsed = time.time() - loop_s
                    to_sleep = max(0, self.frame_interval - elapsed)
                    time.sleep(to_sleep)
        except KeyboardInterrupt:
            log.info('Collection stopped by user')
        finally:
            self.collecting = False
            self.drain_queue()
            self.stop_listeners()
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
            log.info(f'Collection finished. Frames: {self.frame_count}, Events: {len(self.raw_events)}, Location: {loc}')
            return loc

    # Save processed + raw files
    def align_and_save(self, alignment_strategy='nearest', filename_prefix='btd6_data'):
        if self.frame_count == 0:
            raise RuntimeError('No frames collected')
        frame_ts_arr = np.array(self.frame_timestamps)
        frame_actions = [[] for _ in range(self.frame_count)]
        aligned = 0
        for event in self.raw_events:
            ets = event.timestamp
            if alignment_strategy == 'nearest':
                idx = np.searchsorted(frame_ts_arr, ets)
                if idx == 0:
                    fi = 0
                elif idx == len(frame_ts_arr):
                    fi = len(frame_ts_arr) - 1
                else:
                    if abs(frame_ts_arr[idx - 1] - ets) < abs(frame_ts_arr[idx] - ets):
                        fi = idx - 1
                    else:
                        fi = idx
            elif alignment_strategy == 'previous':
                idx = np.searchsorted(frame_ts_arr, ets, side='right') - 1
                fi = max(0, idx)
            else:
                idx = np.searchsorted(frame_ts_arr, ets, side='right') - 1
                fi = max(0, min(idx, len(frame_ts_arr) - 1))
            action = {'type': event.event_type, 'timestamp': event.timestamp, **event.data}
            frame_actions[fi].append(action)
            aligned += 1
        sid = self.current_session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_file = f'{filename_prefix}_processed_{sid}.json'
        processed = {
            'actions': frame_actions,
            'frame_timestamps': self.frame_timestamps,
            'frame_count': self.frame_count,
            'frames_location': self.frames_folder if self.frames_folder else f'btd6_{sid}.mp4',
            'game_region': self.game_region,
            'fps': self.fps,
            'alignment_strategy': alignment_strategy,
            'use_video': self.use_video
        }
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2)
        raw_file = f'{filename_prefix}_raw_{sid}.json'
        rawd = {
            'raw_events': [asdict(e) for e in self.raw_events],
            'frame_timestamps': self.frame_timestamps,
            'frame_count': self.frame_count,
            'frames_location': self.frames_folder if self.frames_folder else f'btd6_{sid}.mp4',
            'game_region': self.game_region,
            'fps': self.fps,
            'use_video': self.use_video
        }
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(rawd, f, indent=2)
        log.info(f'Processed saved: {processed_file}\nRaw saved: {raw_file}')
        return processed_file, raw_file

# -----------------------------
# Dataset (unchanged)
# -----------------------------
def _natural_sort_key(path: str):
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    if m:
        try:
            return (0, int(m.group(1)), base)
        except Exception:
            return (1, base)
    return (1, base)

class BTD6Dataset(Dataset):
    def __init__(self, actions_file, sequence_length=5):
        self.sequence_length = max(1, int(sequence_length))
        with open(actions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.actions = data.get('actions', [])
        self.frame_timestamps = data.get('frame_timestamps', [])
        self.frame_count = int(data.get('frame_count', 0) or 0)
        self.frames_location = data.get('frames_location', '')
        self.game_region = data.get('game_region', None)
        self.fps = data.get('fps', None)
        self.alignment_strategy = data.get('alignment_strategy', 'unknown')
        self.use_video = bool(data.get('use_video', False))
        log.info(f'Dataset: frames_location={self.frames_location} use_video={self.use_video} align={self.alignment_strategy}')

        if self.use_video:
            if not os.path.exists(self.frames_location):
                raise FileNotFoundError(f'Video not found: {self.frames_location}')
            self.cap = cv2.VideoCapture(self.frames_location)
            if not self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
                raise RuntimeError(f'Cannot open video file: {self.frames_location}')
            frame_count_prop = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count_prop <= 0:
                self._len = int(self.frame_count) if self.frame_count > 0 else 0
            else:
                self._len = frame_count_prop
        else:
            if not os.path.isdir(self.frames_location):
                raise FileNotFoundError(f'Frames folder not found: {self.frames_location}')
            frame_paths = [os.path.join(self.frames_location, f) for f in os.listdir(self.frames_location) if f.lower().endswith('.png')]
            self.frame_files = sorted(frame_paths, key=_natural_sort_key)
            if not self.frame_files:
                raise FileNotFoundError(f'No PNGs in {self.frames_location}')
            self._len = len(self.frame_files)

        minlen = min(self._len, len(self.actions), len(self.frame_timestamps))
        if minlen != self._len:
            log.warning(f'truncating to min length {minlen}')
            self._len = minlen
            self.actions = self.actions[:minlen]
            self.frame_timestamps = self.frame_timestamps[:minlen]

        self.labels = self._process_actions()
        self.class_weights = self._calc_class_weights()
        log.info(f'Dataset loaded: {self._len} frames')

    def _get_frame(self, idx):
        if self.use_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError(f'Failed to read frame {idx} from video')
            return frame
        else:
            fp = self.frame_files[idx]
            frame = cv2.imread(fp)
            if frame is None:
                raise RuntimeError(f'Failed to read frame {fp}')
            return frame

    def _process_actions(self):
        labels = []
        for i in range(self._len):
            acts = self.actions[i] if i < len(self.actions) else []
            label = {'action': 0, 'x': 0.0, 'y': 0.0, 'needs_position': False}
            if acts and isinstance(acts, list):
                for a in acts:
                    if not isinstance(a, dict):
                        continue
                    t = a.get('type')
                    if t == 'mouse_click':
                        btn = a.get('button','').lower()
                        label['action'] = 1 if 'left' in btn else 2
                        label['x'] = float(a.get('x', 0.0))
                        label['y'] = float(a.get('y', 0.0))
                        label['needs_position'] = True
                        break
                    elif t == 'key_press':
                        label['action'] = 3
                        label['needs_position'] = False
                    elif t == 'mouse_drag':
                        label['action'] = 4
                        label['x'] = float(a.get('x', 0.0))
                        label['y'] = float(a.get('y', 0.0))
                        label['needs_position'] = True
            labels.append(label)
        return labels

    def _calc_class_weights(self):
        counts = Counter([l['action'] for l in self.labels])
        total = sum(counts.values())
        num_classes = 5
        weights = []
        for i in range(num_classes):
            c = counts.get(i, 1)
            weights.append(total / (num_classes * c) if c > 0 else 0.0)
        log.info('Action counts: ' + ', '.join([f'{i}:{counts.get(i,0)}' for i in range(num_classes)]))
        return torch.FloatTensor(weights)

    def get_sample_weights(self):
        return [self.class_weights[l['action']].item() for l in self.labels]

    def __len__(self):
        return max(0, self._len - self.sequence_length + 1)

    def __getitem__(self, idx):
        seq = []
        for i in range(idx, idx + self.sequence_length):
            if i >= self._len:
                i = self._len - 1
            frame = self._get_frame(i)
            frame = cv2.resize(frame, (320, 180))
            frame = frame.astype(np.float32) / 255.0
            seq.append(frame)
        arr = np.stack(seq, axis=0)
        arr = np.transpose(arr, (0,3,1,2))
        label = self.labels[min(idx + self.sequence_length - 1, self._len - 1)]
        frames_tensor = torch.tensor(arr, dtype=torch.float32)
        action_tensor = torch.tensor(label['action'], dtype=torch.long)
        pos_tensor = torch.tensor([label['x'], label['y']], dtype=torch.float32)
        needs_pos_tensor = torch.tensor(label['needs_position'], dtype=torch.bool)
        return frames_tensor, action_tensor, pos_tensor, needs_pos_tensor

# -----------------------------
# Model (unchanged)
# -----------------------------
class BTD6ImitationNet(nn.Module):
    def __init__(self, num_actions=5, sequence_length=5):
        super(BTD6ImitationNet, self).__init__()
        self.sequence_length = sequence_length
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
            nn.AdaptiveAvgPool2d((4,7))
        )
        self.lstm = nn.LSTM(input_size=128*4*7, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        self.action_head = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_actions))
        self.position_head = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,2), nn.Sigmoid())

    def forward(self, x):
        b = x.size(0)
        seq = x.size(1)
        features = []
        for i in range(seq):
            f = x[:, i, :, :, :]
            out = self.frame_encoder(f)
            out = out.view(b, -1)
            features.append(out)
        feat = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(feat)
        last = lstm_out[:, -1, :]
        logits = self.action_head(last)
        pos = self.position_head(last)
        return logits, pos

# -----------------------------
# Trainer (unchanged)
# -----------------------------
class BTD6Trainer:
    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model.to(device)
        self.device = device
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.action_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.position_criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.action_weight = 1.0
        self.position_weight = 2.0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        action_correct = 0
        total = 0
        pos_err = 0.0
        pos_samples = 0
        for batch_idx, (frames, actions, positions, needs_pos) in enumerate(loader):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            positions = positions.to(self.device)
            needs_pos = needs_pos.to(self.device)

            self.optimizer.zero_grad()
            logits, pred_pos = self.model(frames)
            a_loss = self.action_criterion(logits, actions)
            pos_raw = self.position_criterion(pred_pos, positions)
            pos_per = pos_raw.mean(dim=1)
            masked = pos_per * needs_pos.float()
            num_p = needs_pos.sum().item()
            if num_p > 0:
                pos_loss = masked.sum() / num_p
            else:
                pos_loss = torch.tensor(0.0, device=self.device)
            loss = self.action_weight * a_loss + self.position_weight * pos_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            action_correct += (pred == actions).sum().item()
            total += actions.size(0)
            if num_p > 0:
                pos_err += pos_loss.item() * num_p
                pos_samples += num_p
            if batch_idx % 10 == 0:
                log.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Action: {a_loss.item():.4f}, Pos: {pos_loss.item():.4f}')
        avg_loss = total_loss / max(1, len(loader))
        acc = 100.0 * action_correct / max(1, total)
        avg_pos_err = pos_err / max(1, pos_samples) if pos_samples > 0 else 0
        return avg_loss, acc, avg_pos_err

    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        action_correct = 0
        total = 0
        pos_err = 0.0
        pos_samples = 0
        with torch.no_grad():
            for frames, actions, positions, needs_pos in loader:
                frames = frames.to(self.device)
                actions = actions.to(self.device)
                positions = positions.to(self.device)
                needs_pos = needs_pos.to(self.device)
                logits, pred_pos = self.model(frames)
                a_loss = self.action_criterion(logits, actions)
                pos_raw = self.position_criterion(pred_pos, positions)
                pos_per = pos_raw.mean(dim=1)
                masked = pos_per * needs_pos.float()
                num_p = needs_pos.sum().item()
                if num_p > 0:
                    pos_loss = masked.sum() / num_p
                else:
                    pos_loss = torch.tensor(0.0, device=self.device)
                loss = self.action_weight * a_loss + self.position_weight * pos_loss
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                action_correct += (pred == actions).sum().item()
                total += actions.size(0)
                if num_p > 0:
                    pos_err += pos_loss.item() * num_p
                    pos_samples += num_p
        avg_loss = val_loss / max(1, len(loader))
        acc = 100.0 * action_correct / max(1, total)
        avg_pos_err = pos_err / max(1, pos_samples) if pos_samples > 0 else 0
        return avg_loss, acc, avg_pos_err

    def train(self, train_loader, val_loader, epochs=10, save_path='btd6_model.pth'):
        best_val = float('inf')
        for epoch in range(epochs):
            log.info(f'=== Epoch {epoch+1}/{epochs} ===')
            tr_loss, tr_acc, tr_pos = self.train_epoch(train_loader)
            val_loss, val_acc, val_pos = self.validate(val_loader)
            log.info(f'Train Loss {tr_loss:.4f}, Acc {tr_acc:.2f}%, PosErr {tr_pos:.4f}')
            log.info(f'Val   Loss {val_loss:.4f}, Acc {val_acc:.2f}%, PosErr {val_pos:.4f}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, save_path)
                log.info(f'New best saved: {save_path}')

# -----------------------------
# Inference / Play (unchanged)
# -----------------------------
def run_inference(model_path, game_region, fps=10, sequence_length=5, action_map=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BTD6ImitationNet(num_actions=5, sequence_length=sequence_length)
    ck = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in ck:
        net.load_state_dict(ck['model_state_dict'])
    else:
        net.load_state_dict(ck)
    net.to(DEVICE)
    net.eval()
    log.info('Model loaded for inference')

    # smoothing and cooldowns
    pos_ema = None
    action_cooldowns = {1:0.12, 2:0.12, 3:0.2, 4:0.2}
    last_times = {k:0.0 for k in action_cooldowns}
    from collections import deque
    frame_q = deque(maxlen=sequence_length)

    with mss.mss() as sct:
        # prefill
        first = np.array(sct.grab(game_region))
        if first.shape[2] == 4:
            first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)
        pf = cv2.resize(first, (320,180)).astype(np.float32)/255.0
        for _ in range(sequence_length):
            frame_q.append(pf)
        try:
            while True:
                t0 = time.time()
                img = np.array(sct.grab(game_region))
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                pf = cv2.resize(img, (320,180)).astype(np.float32)/255.0
                frame_q.append(pf)
                seq_np = np.stack(list(frame_q), axis=0)
                seq_np = np.expand_dims(seq_np, axis=0)
                seq_t = torch.tensor(np.transpose(seq_np, (0,1,4,2,3)), dtype=torch.float32, device=DEVICE)
                # Note: model expects shape (B, seq, C, H, W)
                with torch.no_grad():
                    logits, pos = net(seq_t)
                    act = int(logits.argmax(dim=1).cpu().item())
                    pos = pos.cpu().numpy()[0]
                if pos_ema is None:
                    pos_ema = pos
                else:
                    alpha = 0.6
                    pos_ema = alpha * pos + (1-alpha) * pos_ema
                abs_x = int(game_region['left'] + pos_ema[0] * game_region['width'])
                abs_y = int(game_region['top'] + pos_ema[1] * game_region['height'])
                now = time.time()
                performed = False
                if act == 1 and now - last_times[1] > action_cooldowns[1]:
                    pyautogui.click(x=abs_x, y=abs_y, button='left')
                    last_times[1] = now
                    performed = True
                elif act == 2 and now - last_times[2] > action_cooldowns[2]:
                    pyautogui.click(x=abs_x, y=abs_y, button='right')
                    last_times[2] = now
                    performed = True
                elif act == 3 and now - last_times[3] > action_cooldowns[3]:
                    key = action_map.get(3) if action_map else None
                    if key:
                        pyautogui.press(key)
                        last_times[3] = now
                        performed = True
                elif act == 4 and now - last_times[4] > action_cooldowns[4]:
                    pyautogui.moveTo(abs_x, abs_y)
                    pyautogui.dragRel(10, 0, duration=0.05)
                    last_times[4] = now
                    performed = True
                # maintain fps
                elapsed = time.time() - t0
                time.sleep(max(0, 1.0/fps - elapsed))
        except KeyboardInterrupt:
            log.info('Inference stopped by user')
        except pyautogui.FailSafeException:
            log.info('PyAutoGUI failsafe triggered (mouse to top-left)')

# -----------------------------
# CLI
# -----------------------------
def menu():
    print('='*60)
    print('BTD6 - Single-file tool')
    print('1) Collect data')
    print('2) Train model')
    print('3) Play (inference)')
    print('q) Quit')
    print('='*60)
    return input('Choose mode (1/2/3/q): ').strip()

def _read_region_from_input():
    """
    Reads top/left/width/height from user input and returns a dict.
    """
    try:
        top = int(input('top (default 0): ') or '0')
        left = int(input('left (default 0): ') or '0')
        width = int(input('width (default 1600): ') or '1600')
        height = int(input('height (default 900): ') or '900')
    except Exception:
        top, left, width, height = 0, 0, 1600, 900
    return {'top': top, 'left': left, 'width': width, 'height': height}

def choose_region_interactive() -> dict:
    """
    Let user pick region method:
      1) Manual input
      2) Auto-center default region
      3) Detect by window title (Windows only)
    Returns region dict.
    """
    print('Choose capture region method:')
    print(' 1) Manual')
    print(' 2) Auto-center default (1600x900 centered)')
    print(' 3) Detect by window title (Windows only, recommended)')
    choice = input('Method (1/2/3, default 2): ').strip() or '2'
    if choice == '1':
        region = _read_region_from_input()
        return region
    elif choice == '3':
        if not (HAS_PYGETWINDOW or HAS_WIN32):
            print('Window-detection libs not available. Install pygetwindow or pywin32 to use this feature.')
            print('Falling back to auto-center.')
            region = {'top': 0, 'left': 0, 'width': 1600, 'height': 900}
            return center_region_if_needed(region)
        substr = input('Enter a substring of the game window title (e.g. "bloons", "btd6", "Bloons TD"): ').strip()
        found = find_window_region_by_title(substr)
        if found:
            # Some games include the border; optionally allow slight shrink to avoid title bar - we keep raw rect
            return found
        else:
            print('Window not found. Falling back to auto-center.')
            region = {'top': 0, 'left': 0, 'width': 1600, 'height': 900}
            return center_region_if_needed(region)
    else:
        region = {'top': 0, 'left': 0, 'width': 1600, 'height': 900}
        return center_region_if_needed(region)

def run_collect_flow():
    try:
        fps = int(input('FPS (default 5): ') or '5')
    except Exception:
        fps = 5
    storage = input('Storage mode - "frames" or "video" (default frames): ').strip().lower() or 'frames'
    use_video = storage == 'video'
    print('Set game capture region.')
    region = choose_region_interactive()
    print(f'Using capture region: top={region["top"]}, left={region["left"]}, width={region["width"]}, height={region["height"]}')
    try:
        duration = int(input('Duration seconds (default 60): ') or '60')
    except Exception:
        duration = 60
    collector = BTD6DataCollector(game_region=region, fps=fps, use_video=use_video)
    input('Position the game window (if needed) and press Enter to start...')
    print('Collecting in 3 seconds...')
    time.sleep(3)
    frames_loc = collector.collect(duration=duration)
    proc, raw = collector.align_and_save(alignment_strategy='nearest')
    print('Saved:', proc, raw)

def run_train_flow():
    actions_file = input('Processed actions file path (.json): ').strip()
    if not os.path.exists(actions_file):
        print('File not found')
        return
    try:
        seq_len = int(input('Sequence length (default 5): ') or '5')
    except Exception:
        seq_len = 5
    dataset = BTD6Dataset(actions_file, sequence_length=seq_len)
    if len(dataset) < 4:
        print('Dataset too small for training. Collect more data.')
        return
    class_weights = dataset.class_weights
    sample_weights = dataset.get_sample_weights()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    if hasattr(train_dataset, 'indices'):
        train_indices = list(train_dataset.indices)
    else:
        train_indices = list(range(train_size))
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights))
    batch_size = int(input('Batch size (default 8): ') or '8')
    num_workers_default = 0 if os.name == 'nt' else 2
    num_workers = int(input(f'Num workers (default {num_workers_default}): ') or str(num_workers_default))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = BTD6ImitationNet(num_actions=5, sequence_length=seq_len)
    trainer = BTD6Trainer(model, device, class_weights=class_weights)
    epochs = int(input('Epochs (default 20): ') or '20')
    save_path = input('Save path (default btd6_model.pth): ') or 'btd6_model.pth'
    trainer.train(train_loader, val_loader, epochs=epochs, save_path=save_path)
    print('Training done. Model saved to', save_path)

def run_play_flow():
    model_path = input('Model .pth path: ').strip()
    if not os.path.exists(model_path):
        print('Model not found')
        return
    print('Set capture region for inference.')
    region = choose_region_interactive()
    print(f'Using capture region: top={region["top"]}, left={region["left"]}, width={region["width"]}, height={region["height"]}')
    fps = int(input('Infer FPS (default 8): ') or '8')
    seq = int(input('Sequence length (default 5): ') or '5')
    print('Action mapping: action idx 3 is "key press" - set a key or leave empty to disable')
    key_for_3 = input('Key for action 3 (e.g. space): ').strip() or None
    action_map = {3: key_for_3} if key_for_3 else {}
    print('Starting inference in 3 seconds. Put game in front (windowed/borderless).')
    time.sleep(3)
    run_inference(model_path, region, fps=fps, sequence_length=seq, action_map=action_map)

if __name__ == '__main__':
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    # show short info about window-detection capability
    if os.name == 'nt':
        if HAS_PYGETWINDOW or HAS_WIN32:
            log.info('Window-detection: available (pygetwindow or win32api found). Use method 3 in region chooser.')
        else:
            log.info('Window-detection: NOT available. To enable, install "pygetwindow" or "pywin32" (pip).')
    while True:
        choice = menu()
        if choice == '1':
            run_collect_flow()
        elif choice == '2':
            run_train_flow()
        elif choice == '3':
            run_play_flow()
        elif choice.lower() == 'q':
            print('Bye!')
            break
        else:
            print('Invalid choice')
