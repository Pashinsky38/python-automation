"""
BTD6 Imitation Learning Project - FIXED MEMORY & THREADING
==========================================================
Critical fixes implemented:
- Streaming frames to disk (no memory bloat)
- Thread-safe event collection with queue
- Accurate timestamps (captured before operations)
- Chunked JSON writing for large files
- Option for video or individual frame storage

Key improvements:
- Memory efficient: streams frames directly to disk
- Thread-safe: uses queue for event collection
- Accurate: timestamps captured at the right moment
- Robust: handles large datasets without memory issues

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
import threading
from collections import deque, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import shutil
import tempfile

# -----------------------------
# Data collector (FIXED MEMORY & THREADING)
# -----------------------------
@dataclass
class RawEvent:
    """Raw event with timestamp for proper synchronization"""
    timestamp: float
    event_type: str  # 'key_press', 'mouse_click', 'mouse_drag'
    data: dict  # Event-specific data

class BTD6DataCollector:
    """Fixed collector with disk streaming and thread-safe event collection"""

    def __init__(self, game_region=None, fps=10, use_video=False):
        self.game_region = game_region or {"top": 0, "left": 0, "width": 1600, "height": 900}
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.use_video = use_video

        # Thread-safe event queue
        self.event_queue = queue.Queue()
        self.raw_events = []  # Collected from queue in main thread
        
        # Frame storage
        self.frame_timestamps = []
        self.frame_count = 0
        self.frames_folder = None
        self.video_writer = None
        self.current_session_id = None
        
        # Current mouse position (protected by lock)
        self.mouse_lock = threading.Lock()
        self.current_mouse_pos = (0, 0)

        # Input tracking
        self.collecting = False

        # Input listeners
        self.keyboard_listener = None
        self.mouse_listener = None

        print("FIXED BTD6 Data Collector initialized")
        print(f"Game region: {self.game_region}")
        print(f"Capture rate: {fps} FPS")
        print(f"Storage mode: {'video' if use_video else 'individual frames'}")
        print("✓ Memory-efficient streaming enabled")
        print("✓ Thread-safe event collection enabled")

    def on_key_press(self, key):
        if not self.collecting:
            return
        
        # Capture timestamp immediately
        timestamp = time.time()
        
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except Exception:
            key_name = str(key)
        
        # Thread-safe read of mouse position
        with self.mouse_lock:
            mouse_x, mouse_y = self.current_mouse_pos
        
        event = RawEvent(
            timestamp=timestamp,
            event_type='key_press',
            data={
                'key': key_name,
                'mouse_x': mouse_x,
                'mouse_y': mouse_y
            }
        )
        # Thread-safe queue put
        self.event_queue.put(event)

    def on_mouse_move(self, x, y):
        """Track mouse movement"""
        if not self.collecting:
            return
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            # Thread-safe update
            with self.mouse_lock:
                self.current_mouse_pos = (rel_x, rel_y)

    def on_mouse_click(self, x, y, button, pressed):
        if not self.collecting or not pressed:
            return
        
        # Capture timestamp immediately
        timestamp = time.time()
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']

        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            event = RawEvent(
                timestamp=timestamp,
                event_type='mouse_click',
                data={
                    'x': float(rel_x) / self.game_region['width'],
                    'y': float(rel_y) / self.game_region['height'],
                    'button': str(button),
                    'abs_x': rel_x,
                    'abs_y': rel_y
                }
            )
            # Thread-safe queue put
            self.event_queue.put(event)

    def on_mouse_drag(self, x, y):
        """Track mouse drag for camera movement"""
        if not self.collecting:
            return
        
        # Capture timestamp immediately
        timestamp = time.time()
        rel_x = x - self.game_region['left']
        rel_y = y - self.game_region['top']
        
        if 0 <= rel_x < self.game_region['width'] and 0 <= rel_y < self.game_region['height']:
            event = RawEvent(
                timestamp=timestamp,
                event_type='mouse_drag',
                data={
                    'x': float(rel_x) / self.game_region['width'],
                    'y': float(rel_y) / self.game_region['height'],
                    'abs_x': rel_x,
                    'abs_y': rel_y
                }
            )
            # Thread-safe queue put
            self.event_queue.put(event)

    def start_input_listeners(self):
        """Start input listeners"""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
            on_move=self.on_mouse_move
        )
        self.keyboard_listener.start()
        self.mouse_listener.start()
        print("Input listeners started (thread-safe)")

    def stop_input_listeners(self):
        """Stop input listeners"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        print("Input listeners stopped")

    def init_frame_storage(self):
        """Initialize frame storage (video or folder)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = timestamp
        
        if self.use_video:
            # Initialize video writer
            video_file = f"btd6_frames_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_file,
                fourcc,
                self.fps,
                (self.game_region['width'], self.game_region['height'])
            )
            print(f"Video writer initialized: {video_file}")
            return video_file
        else:
            # Create folder for frames
            self.frames_folder = f"btd6_frames_{timestamp}"
            os.makedirs(self.frames_folder, exist_ok=True)
            print(f"Frames folder created: {self.frames_folder}")
            return self.frames_folder

    def save_frame(self, frame, frame_idx):
        """Save a single frame to disk (streaming)"""
        if self.use_video and self.video_writer:
            self.video_writer.write(frame)
        elif self.frames_folder:
            path = os.path.join(self.frames_folder, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(path, frame)

    def cleanup_frame_storage(self):
        """Clean up frame storage"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("Video writer released")

    def capture_and_save_frame(self, sct, frame_idx):
        """Capture a frame and immediately save to disk"""
        try:
            # Capture timestamp BEFORE grabbing frame
            timestamp = time.time()
            
            screenshot = sct.grab(self.game_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Stream to disk immediately (no memory storage)
            self.save_frame(frame, frame_idx)
            
            return timestamp
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

    def drain_event_queue(self):
        """Drain event queue in main thread (thread-safe)"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self.raw_events.append(event)
            except queue.Empty:
                break

    def collect_data(self, duration=300):
        """Collect data with disk streaming and thread-safe events"""
        print(f"Starting data collection for {duration} seconds...")
        print("Frames will be streamed to disk (no memory bloat)")
        print("Events collected via thread-safe queue")
        print("Move mouse to upper-left corner to trigger PyAutoGUI fail-safe.")
        
        # Initialize storage
        frames_location = self.init_frame_storage()
        
        self.collecting = True
        self.frame_count = 0
        self.frame_timestamps = []
        self.raw_events = []
        
        self.start_input_listeners()
        start_time = time.time()

        try:
            with mss.mss() as sct:
                while self.collecting and (time.time() - start_time) < duration:
                    loop_start = time.time()
                    
                    # Capture and stream frame to disk
                    timestamp = self.capture_and_save_frame(sct, self.frame_count)
                    
                    if timestamp is not None:
                        self.frame_timestamps.append(timestamp)
                        self.frame_count += 1
                        
                        # Drain event queue periodically (thread-safe)
                        self.drain_event_queue()
                        
                        if self.frame_count % 50 == 0:
                            print(f"Captured {self.frame_count} frames, {len(self.raw_events)} events...")

                    # Maintain FPS
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.frame_interval - elapsed)
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        finally:
            self.collecting = False
            # Final drain of event queue
            self.drain_event_queue()
            self.stop_input_listeners()
            self.cleanup_frame_storage()
            
            print(f"Collection complete:")
            print(f"  - Frames: {self.frame_count} (streamed to disk)")
            print(f"  - Events: {len(self.raw_events)}")
            print(f"  - Location: {frames_location}")
            if self.frame_timestamps:
                print(f"  - Duration: {self.frame_timestamps[-1] - self.frame_timestamps[0]:.1f}s")

    def align_events_to_frames(self, alignment_strategy='nearest'):
        """Post-process to align events to frames based on timestamps."""
        if not self.frame_timestamps:
            print("No frames to align to!")
            return []
        
        print(f"\nAligning {len(self.raw_events)} events to {self.frame_count} frames...")
        print(f"Strategy: {alignment_strategy}")
        
        # Initialize frame actions list
        frame_actions = [[] for _ in range(self.frame_count)]
        
        # Convert to numpy array for efficient searching
        frame_ts_array = np.array(self.frame_timestamps)
        
        # Process each event
        aligned_count = 0
        out_of_range_count = 0
        
        for event in self.raw_events:
            event_ts = event.timestamp
            
            if alignment_strategy == 'nearest':
                # Find nearest frame
                idx = np.searchsorted(frame_ts_array, event_ts)
                if idx == 0:
                    frame_idx = 0
                elif idx == len(frame_ts_array):
                    frame_idx = len(frame_ts_array) - 1
                else:
                    # Choose between idx-1 and idx based on which is closer
                    if abs(frame_ts_array[idx-1] - event_ts) < abs(frame_ts_array[idx] - event_ts):
                        frame_idx = idx - 1
                    else:
                        frame_idx = idx
                        
            elif alignment_strategy == 'previous':
                # Assign to previous frame (accounts for reaction time)
                idx = np.searchsorted(frame_ts_array, event_ts, side='right') - 1
                frame_idx = max(0, idx)
                
            elif alignment_strategy == 'interval':
                # Assign to frame if event falls in its interval
                idx = np.searchsorted(frame_ts_array, event_ts, side='right') - 1
                frame_idx = max(0, min(idx, len(frame_ts_array) - 1))
            
            else:
                raise ValueError(f"Unknown alignment strategy: {alignment_strategy}")
            
            # Check if event is too far from any frame (potential outlier)
            time_diff = abs(frame_ts_array[frame_idx] - event_ts)
            if time_diff > 1.0:  # More than 1 second away
                out_of_range_count += 1
                print(f"  Warning: Event {time_diff:.2f}s away from nearest frame")
            
            # Convert to the format expected by the rest of the pipeline
            action_dict = {
                'type': event.event_type,
                'timestamp': event.timestamp,
                **event.data
            }
            
            frame_actions[frame_idx].append(action_dict)
            aligned_count += 1
        
        print(f"✓ Aligned {aligned_count} events")
        if out_of_range_count > 0:
            print(f"  ⚠ {out_of_range_count} events were >1s from nearest frame")
        
        # Show distribution of actions per frame
        actions_per_frame = [len(actions) for actions in frame_actions]
        if actions_per_frame:
            print(f"  Actions per frame: min={min(actions_per_frame)}, "
                  f"max={max(actions_per_frame)}, avg={np.mean(actions_per_frame):.2f}")
        
        return frame_actions

    def save_json_chunked(self, data, filename, chunk_size=1000):
        """Save large JSON data in chunks to avoid memory issues"""
        with open(filename, 'w') as f:
            f.write('{\n')
            
            first_key = True
            for key, value in data.items():
                if not first_key:
                    f.write(',\n')
                first_key = False
                
                f.write(f'  "{key}": ')
                
                if isinstance(value, list) and len(value) > chunk_size:
                    # Write large lists in chunks
                    f.write('[\n')
                    for i in range(0, len(value), chunk_size):
                        chunk = value[i:i+chunk_size]
                        chunk_json = json.dumps(chunk, indent=2)[1:-1]  # Remove outer brackets
                        if i > 0:
                            f.write(',\n')
                        f.write(chunk_json)
                    f.write('\n  ]')
                else:
                    json.dump(value, f, indent=2)
            
            f.write('\n}\n')

    def save_data(self, filename_prefix="btd6_data", alignment_strategy='nearest'):
        """Save data with proper timestamp alignment"""
        if self.frame_count == 0:
            raise RuntimeError("No frames collected. Nothing to save.")

        # First, align events to frames
        aligned_actions = self.align_events_to_frames(alignment_strategy)

        # Frame location (already saved during collection)
        frames_location = self.frames_folder if self.frames_folder else f"btd6_frames_{self.current_session_id}.mp4"
        
        print(f"Frames location: {frames_location}")

        # Save processed data (aligned) - use chunked writing for large datasets
        processed_file = f"{filename_prefix}_processed_{self.current_session_id}.json"
        processed_data = {
            'actions': aligned_actions,
            'frame_timestamps': self.frame_timestamps,
            'frame_count': self.frame_count,
            'frames_location': frames_location,
            'game_region': self.game_region,
            'fps': self.fps,
            'alignment_strategy': alignment_strategy,
            'use_video': self.use_video
        }
        
        if len(aligned_actions) > 10000:
            print("Using chunked JSON writing for large dataset...")
            self.save_json_chunked(processed_data, processed_file)
        else:
            with open(processed_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
        
        print(f"Processed data saved to: {processed_file}")

        # Also save raw data for potential re-processing
        raw_file = f"{filename_prefix}_raw_{self.current_session_id}.json"
        raw_data = {
            'raw_events': [asdict(e) for e in self.raw_events],
            'frame_timestamps': self.frame_timestamps,
            'frame_count': self.frame_count,
            'frames_location': frames_location,
            'game_region': self.game_region,
            'fps': self.fps,
            'use_video': self.use_video
        }
        
        if len(self.raw_events) > 10000:
            print("Using chunked JSON writing for raw data...")
            self.save_json_chunked(raw_data, raw_file)
        else:
            with open(raw_file, 'w') as f:
                json.dump(raw_data, f, indent=2)
        
        print(f"Raw data saved to: {raw_file}")

        return frames_location, processed_file, raw_file


# -----------------------------
# Dataset (memory-efficient loading)
# -----------------------------
class BTD6Dataset(Dataset):
    """Dataset with memory-efficient frame loading"""

    def __init__(self, actions_file, sequence_length=5):
        self.sequence_length = sequence_length

        # Load action data
        with open(actions_file, 'r') as f:
            data = json.load(f)
        
        self.actions = data.get('actions', [])
        self.frame_timestamps = data.get('frame_timestamps', [])
        self.frame_count = data.get('frame_count', 0)
        self.frames_location = data.get('frames_location', '')
        self.game_region = data.get('game_region', None)
        self.fps = data.get('fps', None)
        self.alignment_strategy = data.get('alignment_strategy', 'unknown')
        self.use_video = data.get('use_video', False)
        
        print(f"Dataset using alignment strategy: {self.alignment_strategy}")
        print(f"Frames location: {self.frames_location}")
        print(f"Frame storage type: {'video' if self.use_video else 'individual frames'}")

        # Setup frame access
        if self.use_video:
            # Video mode
            if not os.path.exists(self.frames_location):
                raise FileNotFoundError(f"Video file not found: {self.frames_location}")
            self.cap = cv2.VideoCapture(self.frames_location)
            self._len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            # Individual frames mode
            if not os.path.isdir(self.frames_location):
                raise FileNotFoundError(f"Frames folder not found: {self.frames_location}")
            self.frame_files = sorted([
                os.path.join(self.frames_location, f) 
                for f in os.listdir(self.frames_location) 
                if f.endswith('.png')
            ])
            self._len = len(self.frame_files)

        # Ensure consistent lengths
        minlen = min(self._len, len(self.actions), len(self.frame_timestamps))
        if minlen != self._len:
            print(f"Warning: truncating to min length: {minlen}")
            self._len = minlen
            self.actions = self.actions[:minlen]
            self.frame_timestamps = self.frame_timestamps[:minlen]

        # Process labels
        self.labels = self._process_enhanced_actions()
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()
        
        print(f"Dataset loaded: {self._len} frames")
        if self.frame_timestamps:
            print(f"Timestamp range: {self.frame_timestamps[0]:.2f} to {self.frame_timestamps[-1]:.2f}")
            print(f"Average FPS: {self._len / (self.frame_timestamps[-1] - self.frame_timestamps[0]):.2f}")

    def _get_frame(self, idx):
        """Load a single frame from disk (memory-efficient)"""
        if self.use_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from video")
            return frame
        else:
            frame_path = self.frame_files[idx]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise RuntimeError(f"Failed to read frame from {frame_path}")
            return frame

    def _process_enhanced_actions(self):
        """Process properly aligned actions into labels"""
        labels = []
        
        for i in range(self._len):
            frame_actions = self.actions[i] if i < len(self.actions) else []
            
            # Enhanced label format
            label = {'action': 0, 'x': 0.0, 'y': 0.0, 'needs_position': False}
            
            if frame_actions and isinstance(frame_actions, list):
                # Use the first action in the frame (could also aggregate)
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
        
        weights = []
        for i in range(num_classes):
            count = action_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)
        
        print("Action distribution (after alignment):")
        action_names = ['none', 'left_click', 'right_click', 'key', 'drag']
        for i in range(num_classes):
            count = action_counts.get(i, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {action_names[i]}: {count} ({percentage:.1f}%)")
        
        return torch.FloatTensor(weights)

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler"""
        weights = []
        for label in self.labels:
            action = label['action']
            weights.append(self.class_weights[action].item())
        return weights

    def __len__(self):
        return max(0, self._len - self.sequence_length + 1)

    def __getitem__(self, idx):
        frames_seq = []
        
        for i in range(idx, idx + self.sequence_length):
            if i >= self._len:
                i = self._len - 1
            
            # Load frame from disk (memory-efficient)
            frame = self._get_frame(i)
            frame = cv2.resize(frame, (320, 180))
            frame = frame.astype(np.float32) / 255.0
            frames_seq.append(frame)
        
        frames_array = np.stack(frames_seq, axis=0)
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))
        
        label = self.labels[min(idx + self.sequence_length - 1, self._len - 1)]
        
        frames_tensor = torch.tensor(frames_array, dtype=torch.float32)
        action_tensor = torch.tensor(label['action'], dtype=torch.long)
        pos_tensor = torch.tensor([label['x'], label['y']], dtype=torch.float32)
        needs_pos_tensor = torch.tensor(label['needs_position'], dtype=torch.bool)
        
        return frames_tensor, action_tensor, pos_tensor, needs_pos_tensor

    def __del__(self):
        """Clean up video capture if used"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


# -----------------------------
# Post-Processing Tool
# -----------------------------
class EventAlignmentProcessor:
    """Tool to re-process raw data with different alignment strategies"""
    
    @staticmethod
    def reprocess_raw_data(raw_file_path, new_strategy='nearest', output_file=None):
        """Re-align raw events with a different strategy without re-recording."""
        print(f"Loading raw data from: {raw_file_path}")
        with open(raw_file_path, 'r') as f:
            raw_data = json.load(f)
        
        frame_timestamps = raw_data['frame_timestamps']
        raw_events = [RawEvent(**e) for e in raw_data['raw_events']]
        frame_count = raw_data.get('frame_count', len(frame_timestamps))
        
        print(f"Loaded {len(raw_events)} events and {frame_count} frames")
        print(f"Re-aligning with strategy: {new_strategy}")
        
        # Re-create aligned actions
        frame_actions = [[] for _ in range(frame_count)]
        frame_ts_array = np.array(frame_timestamps)
        
        for event in raw_events:
            event_ts = event.timestamp
            
            if new_strategy == 'nearest':
                idx = np.searchsorted(frame_ts_array, event_ts)
                if idx == 0:
                    frame_idx = 0
                elif idx == len(frame_ts_array):
                    frame_idx = len(frame_ts_array) - 1
                else:
                    if abs(frame_ts_array[idx-1] - event_ts) < abs(frame_ts_array[idx] - event_ts):
                        frame_idx = idx - 1
                    else:
                        frame_idx = idx
                        
            elif new_strategy == 'previous':
                idx = np.searchsorted(frame_ts_array, event_ts, side='right') - 1
                frame_idx = max(0, idx)
                
            elif new_strategy == 'interval':
                idx = np.searchsorted(frame_ts_array, event_ts, side='right') - 1
                frame_idx = max(0, min(idx, len(frame_ts_array) - 1))
                
            elif new_strategy == 'delayed':
                # Account for reaction delay (e.g., 200ms)
                REACTION_DELAY = 0.2
                adjusted_ts = event_ts + REACTION_DELAY
                idx = np.searchsorted(frame_ts_array, adjusted_ts)
                frame_idx = min(idx, len(frame_ts_array) - 1)
            
            else:
                raise ValueError(f"Unknown strategy: {new_strategy}")
            
            action_dict = {
                'type': event.event_type,
                'timestamp': event.timestamp,
                **event.data
            }
            frame_actions[frame_idx].append(action_dict)
        
        # Save reprocessed data
        if output_file is None:
            base_name = os.path.splitext(raw_file_path)[0]
            output_file = f"{base_name}_reprocessed_{new_strategy}.json"
        
        processed_data = {
            'actions': frame_actions,
            'frame_timestamps': frame_timestamps,
            'frame_count': frame_count,
            'frames_location': raw_data.get('frames_location', ''),
            'game_region': raw_data['game_region'],
            'fps': raw_data['fps'],
            'alignment_strategy': new_strategy,
            'use_video': raw_data.get('use_video', False)
        }
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"✓ Reprocessed data saved to: {output_file}")
        
        # Show statistics
        actions_per_frame = [len(actions) for actions in frame_actions]
        total_actions = sum(actions_per_frame)
        frames_with_actions = sum(1 for a in actions_per_frame if a > 0)
        
        print(f"\nStatistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total actions: {total_actions}")
        print(f"  Frames with actions: {frames_with_actions} ({frames_with_actions/frame_count*100:.1f}%)")
        if actions_per_frame:
            print(f"  Actions per frame: min={min(actions_per_frame)}, max={max(actions_per_frame)}, avg={np.mean(actions_per_frame):.2f}")
        
        return output_file


# -----------------------------
# Model Architecture (unchanged)
# -----------------------------
class BTD6ImitationNet(nn.Module):
    """Network with temporal processing and position regression"""

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
            nn.AdaptiveAvgPool2d((4, 7))
        )
        
        self.lstm = nn.LSTM(
            input_size=128 * 4 * 7,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
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
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        frame_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            features = self.frame_encoder(frame)
            features = features.view(batch_size, -1)
            frame_features.append(features)
        
        frame_features = torch.stack(frame_features, dim=1)
        lstm_out, _ = self.lstm(frame_features)
        last_output = lstm_out[:, -1, :]
        
        action_logits = self.action_head(last_output)
        positions = self.position_head(last_output)
        
        return action_logits, positions


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
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        self.action_weight = 1.0
        self.position_weight = 2.0

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
            action_loss = self.action_criterion(action_logits, actions)
            
            pos_loss_raw = self.position_criterion(pred_positions, positions)
            pos_loss_per_sample = pos_loss_raw.mean(dim=1)
            masked_pos_loss = pos_loss_per_sample * needs_pos.float()
            
            num_pos_samples = needs_pos.sum().item()
            if num_pos_samples > 0:
                position_loss = masked_pos_loss.sum() / num_pos_samples
            else:
                position_loss = torch.tensor(0.0, device=self.device)
            
            loss = self.action_weight * action_loss + self.position_weight * position_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                      f'Position Loss: {position_loss.item():.4f}')

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
                action_loss = self.action_criterion(action_logits, actions)
                
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
        print("Starting training with memory-efficient dataset...")
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
# Main CLI
# -----------------------------
def main():
    pyautogui.PAUSE = 0.01
    pyautogui.FAILSAFE = True

    print("=" * 60)
    print("BTD6 AI Imitation Learning - FIXED MEMORY & THREADING")
    print("=" * 60)
    print("Critical fixes implemented:")
    print("✓ Memory-efficient: streams frames to disk")
    print("✓ Thread-safe: event collection via queue")
    print("✓ Accurate timestamps")
    print("✓ Chunked JSON writing for large files")
    print("✓ Choice of video or individual frame storage")
    print("=" * 60)

    mode = input("\nChoose mode:\n1. Collect data\n2. Re-align existing data\n3. Train model\n4. Play with model\nEnter choice (1/2/3/4): ").strip()

    if mode == '1':
        print("\n--- Data Collection Mode ---")
        fps = int(input("FPS (default 5, recommended 5-10): ") or "5")
        
        storage_mode = input("Storage mode - video or frames (default frames): ").strip().lower()
        use_video = (storage_mode == 'video')
        
        print("\nAlignment strategies:")
        print("  nearest: Event to nearest frame (default)")
        print("  previous: Event to previous frame (reaction time)")
        print("  interval: Event to containing interval")
        strategy = input("Alignment strategy (default 'nearest'): ").strip() or "nearest"
        
        print("\nPreparing to collect data...")
        print("1. Start BTD6 and enter a game")
        print("2. Position the game window")
        print("3. Press Enter when ready")
        input("\nPress Enter to start collection...")
        
        collector = BTD6DataCollector(fps=fps, use_video=use_video)
        duration = int(input("Enter collection duration in seconds (default 300): ") or "300")
        
        print("\nCollection starting in 3 seconds...")
        time.sleep(3)
        
        collector.collect_data(duration)
        frames_location, processed_file, raw_file = collector.save_data(alignment_strategy=strategy)
        
        print("\n✓ Data collection complete!")
        print(f"Frames: {frames_location}")
        print(f"Processed (aligned): {processed_file}")
        print(f"Raw (for re-processing): {raw_file}")

    elif mode == '2':
        print("\n--- Re-alignment Mode ---")
        raw_file = input("Enter raw data file path (.json): ").strip()
        
        if not os.path.exists(raw_file):
            print("❌ Raw file not found!")
            return
        
        print("\nAlignment strategies:")
        print("  nearest: Event to nearest frame")
        print("  previous: Event to previous frame")
        print("  interval: Event to containing interval")
        print("  delayed: Account for reaction delay")
        
        strategy = input("New alignment strategy: ").strip()
        
        processor = EventAlignmentProcessor()
        output_file = processor.reprocess_raw_data(raw_file, strategy)
        
        print(f"\n✓ Re-alignment complete!")
        print(f"New processed file: {output_file}")

    elif mode == '3':
        print("\n--- Training Mode ---")
        actions_file = input("Enter processed actions file (.json): ").strip()
        
        if not os.path.exists(actions_file):
            print("❌ Actions file not found!")
            return

        print("\nLoading dataset (memory-efficient)...")
        sequence_length = int(input("Sequence length (default 5): ") or "5")
        dataset = BTD6Dataset(actions_file, sequence_length=sequence_length)
        
        # Class balancing
        class_weights = dataset.class_weights
        sample_weights = dataset.get_sample_weights()
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Weighted sampler for balanced training
        train_indices = train_dataset.indices
        train_weights = [sample_weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        
        # Data loaders
        batch_size = int(input("Batch size (default 8): ") or "8")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = BTD6ImitationNet(num_actions=5, sequence_length=sequence_length)
        trainer = BTD6Trainer(model, device, class_weights=class_weights)
        
        epochs = int(input("Number of epochs (default 30): ") or "30")
        trainer.train(train_loader, val_loader, epochs, save_path='best_btd6_model_fixed.pth')
        
        print("\n✓ Training complete!")

    else:
        print("❌ Invalid choice.")


if __name__ == "__main__":
    main()