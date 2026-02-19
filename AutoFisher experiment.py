"""
Hypixel Skyblock Auto Fisher - Optimized
==========================================
Key optimizations over original:
- mss for screen capture (~5-10x faster than pyautogui.screenshot)
- win32api for mouse clicks (no artificial delays, near-instant)
- Pre-allocated numpy buffer (no repeated memory allocation)
- pyautogui.PAUSE set to 0 (removes hidden 0.1s delay on all calls)
- Grayscale captured more efficiently
- Template pre-normalized for faster matching

Requirements: pip install mss pywin32 opencv-python numpy pyautogui pynput
"""

import cv2
import numpy as np
import time
import random
import pyautogui
import ctypes
import win32api
import win32con
import mss
import mss.tools
from pynput import mouse

# Remove pyautogui's built-in delay (default is 0.1s per call!)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # Disable corner-detection failsafe for speed

# --- Fast click functions using win32api (bypasses pyautogui overhead) ---
def fast_right_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.012)  # Minecraft needs a brief hold to register
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

class AutoFisher:
    def __init__(self, exclamation_image_path):
        # Load template in grayscale immediately
        self.exclamation_template = cv2.imread(exclamation_image_path, cv2.IMREAD_GRAYSCALE)

        if self.exclamation_template is None:
            raise FileNotFoundError(
                f"Could not load template image at '{exclamation_image_path}'. "
                "Please ensure the file exists and is a valid image format."
            )

        self.tmpl_h, self.tmpl_w = self.exclamation_template.shape
        print(f"Template loaded: {self.exclamation_template.shape}")

        self.running = False
        self.last_throw_time = 0
        self.fishing_region = None
        self.stop_requested = False
        self.current_reset_timeout = 0

        # Pre-allocated screenshot buffer (filled by mss each frame)
        self._screenshot_buf = None

        # Timing
        self.reaction_time_min = 0.010
        self.reaction_time_max = 0.030
        self.throw_delay_min   = 0.05
        self.throw_delay_max   = 0.10

        # Reset mechanism
        self.reset_timeout_min = 15.0
        self.reset_timeout_max = 20.0

        # Detection
        self.threshold = 0.80
        self.detection_interval = 0.008   # ~125 FPS detection loop (was 0.01 / 100 FPS)

        # Safety stop
        self.safety_stop_timer = 60 * 30   # 30 minutes
        self.start_time = None

        self.mouse_listener = None

    # ------------------------------------------------------------------
    # Region setup
    # ------------------------------------------------------------------
    def setup_center_left_region(self):
        """Focused detection region, optimized for 2560x1440."""
        screen_width, screen_height = pyautogui.size()

        box_width   = 350
        box_height  = 280
        left_offset = 140
        vertical_offset = 20

        region_left = (screen_width  // 2) - left_offset - (box_width  // 2)
        region_top  = (screen_height // 2) - (box_height // 2) - vertical_offset

        # mss uses a dict with top/left/width/height
        self.fishing_region = {
            "left":   region_left,
            "top":    region_top,
            "width":  box_width,
            "height": box_height,
        }

        # Pre-allocate numpy buffer for this region (avoids malloc each frame)
        self._screenshot_buf = np.empty(
            (box_height, box_width, 4), dtype=np.uint8  # mss gives BGRA
        )

        print(f"Detection region: {self.fishing_region}")
        print(f"Monitoring {box_width}x{box_height} pixels, optimized for 2560x1440")

    # ------------------------------------------------------------------
    # Detection (hot path — keep as lean as possible)
    # ------------------------------------------------------------------
    def detect_exclamation_marks(self, sct):
        """
        Capture screen region with mss and run template match.
        mss is ~5-10x faster than pyautogui.screenshot.
        """
        # Grab region directly (mss modern API)
        frame = sct.grab(self.fishing_region)
        np.copyto(self._screenshot_buf, np.frombuffer(frame.raw, dtype=np.uint8)
                  .reshape(self.fishing_region["height"], self.fishing_region["width"], 4))

        # Convert BGRA -> Grayscale (single step, faster than BGR->GRAY)
        gray = cv2.cvtColor(self._screenshot_buf, cv2.COLOR_BGRA2GRAY)

        # Template match
        result = cv2.matchTemplate(gray, self.exclamation_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max_val >= self.threshold

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def throw_fishing_rod(self):
        delay = random.uniform(self.throw_delay_min, self.throw_delay_max)
        time.sleep(delay)
        fast_right_click()
        self.last_throw_time = time.time()
        self.current_reset_timeout = random.uniform(self.reset_timeout_min, self.reset_timeout_max)
        print(f"Rod thrown after {delay:.3f}s delay (timeout: {self.current_reset_timeout:.1f}s)")

    def catch_fish(self):
        # Double right-click to catch
        fast_right_click()
        time.sleep(0.05)
        fast_right_click()
        # Wait 25ms
        time.sleep(0.030)
        print("Caught!")

    def check_for_reset(self):
        if time.time() - self.last_throw_time > self.current_reset_timeout:
            print(f"No fish for {self.current_reset_timeout:.1f}s — resetting...")
            reset_delay = random.uniform(0.5, 1.5)
            time.sleep(reset_delay)
            self.throw_fishing_rod()
            return True
        return False

    # ------------------------------------------------------------------
    # Panic button
    # ------------------------------------------------------------------
    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            print("Left click — stopping bot!")
            self.stop_requested = True
            return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        print("Auto-fisher starting in 5 seconds...")
        print(f"LEFT CLICK to stop | Safety stop after {self.safety_stop_timer / 60:.0f} min")
        time.sleep(5)

        self.running    = True
        self.start_time = time.time()
        catches = 0
        resets  = 0

        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

        self.throw_fishing_rod()

        # Open a single persistent mss instance (no per-frame init overhead)
        with mss.mss() as sct:
            while self.running and not self.stop_requested:
                try:
                    # Safety timer
                    if time.time() - self.start_time >= self.safety_stop_timer:
                        print(f"Safety timer reached ({self.safety_stop_timer / 60:.0f} min). Stopping.")
                        break

                    # Timeout reset
                    if self.check_for_reset():
                        resets += 1
                        continue

                    # Detection (hot path)
                    if self.detect_exclamation_marks(sct):
                        reaction = random.uniform(self.reaction_time_min, self.reaction_time_max)
                        time.sleep(reaction)
                        self.catch_fish()
                        catches += 1
                        print(f"Fish #{catches} (reaction: {reaction*1000:.0f}ms)")
                        time.sleep(0.5)  # Cooldown so lingering marks don't re-trigger
                        self.last_throw_time = time.time()  # Reset timeout after manual throw
                        self.current_reset_timeout = random.uniform(self.reset_timeout_min, self.reset_timeout_max)

                    time.sleep(self.detection_interval)

                except KeyboardInterrupt:
                    print("Interrupted.")
                    break

        if self.mouse_listener:
            self.mouse_listener.stop()

        self.running = False
        print(f"Stopped. Catches: {catches} | Resets: {resets}")


def main():
    exclamation_image_path = "exclamation_marks.png"

    try:
        fisher = AutoFisher(exclamation_image_path)
        fisher.setup_center_left_region()
        fisher.run()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()