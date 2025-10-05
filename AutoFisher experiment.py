"""
Hypixel Skyblock Auto Fisher
=============================

A computer vision-based auto-fishing bot for Hypixel Skyblock that detects the 
three exclamation marks (!!!) that appear above the bobber when a fish is caught.

How it works:
- Uses OpenCV template matching to detect the exclamation marks pattern
- Performs a right-click when detected to catch the fish
- Automatically throws the rod again after catching
- Includes reset mechanism if no fish is caught within timeout period
- Adds human-like reaction delays and random behaviors to appear more natural
- Safety timer automatically stops the bot after 1 hour

Setup:
1. Take a screenshot of the three exclamation marks that appear when fishing
2. Save it as "exclamation_marks.png" in the same directory as this script
3. Run the script
4. The bot will start fishing after a 5-second countdown

Note on Detection Speed:
Due to server lag in Hypixel, players get visual indicators before the fish 
actually arrives, allowing them to prepare for the click. Combined with the 
exclamation marks appearing on screen, fast reaction times (30-100ms) are 
achievable by skilled players who are ready to click. This bot simulates that
prepared, fast reaction rather than superhuman reflexes.

Controls:
- LEFT CLICK: Stop the bot at any time
- Automatic safety stop after 1 hour
"""

import pyautogui
import cv2
import numpy as np
import time
import random
from pynput import mouse

class AutoFisher:
    def __init__(self, exclamation_image_path):
        # Load and validate the template image
        self.exclamation_template = cv2.imread(exclamation_image_path, 0)
        
        if self.exclamation_template is None:
            raise FileNotFoundError(
                f"Could not load template image at '{exclamation_image_path}'. "
                "Please ensure the file exists and is a valid image format."
            )
        
        print(f"Template loaded successfully: {self.exclamation_template.shape}")
        
        self.running = False
        self.last_throw_time = 0
        self.fishing_region = None
        self.stop_requested = False
        
        # Human-like timing parameters
        self.reaction_time_min = 0.03  # Minimum reaction time
        self.reaction_time_max = 0.1   # Maximum reaction time
        self.throw_delay_min = 0.1     # Min delay before throwing again
        self.throw_delay_max = 0.3     # Max delay before throwing again
        
        # Reset mechanism
        self.reset_timeout_min = 15.0  # Minimum time before reset
        self.reset_timeout_max = 20.0  # Maximum time before reset
        
        # Detection threshold
        self.threshold = 0.8  #  Change accordingly if needed
        
        # Fast detection settings
        self.detection_interval = 0.03  # Check every 30ms for detection
        
        # Behavior probabilities (adjust these to tune randomness)
        self.break_chance = 0.00       # 1% chance of taking a short break NOT USED CURRENTLY
        
        # Safety stop timer (in seconds)
        self.safety_stop_timer = 60 * 15  # minutes * seconds
        self.start_time = None
        
        # Setup mouse listener for panic button
        self.mouse_listener = None
        
    def on_click(self, x, y, button, pressed):
        """Handle mouse click events for panic button"""
        if pressed and button == mouse.Button.left:
            print("Left click detected - stopping bot!")
            self.stop_requested = True
            return False  # Stop listener
        
    def setup_center_left_region(self):
        """Set a small focused region slightly left of screen center for exclamation marks"""
        screen_width, screen_height = pyautogui.size()
        
        # Create a focused box for the exclamation marks
        # Optimized for 2560x1440 resolution where marks appear ~140 pixels left of center
        box_width = 350   # Slightly wider for better coverage
        box_height = 280  # Good height for bobber movement
        
        # Position: left of center where exclamation marks appear
        # For 2560x1440: center is 1280, marks at ~1140, so offset by 140
        left_offset = 140  # Pixels left from center (optimized for your resolution)
        region_left = (screen_width // 2) - left_offset - (box_width // 2)
        
        # Vertically: slightly above center where bobber typically floats
        vertical_offset = 20  # Slightly above perfect center
        region_top = (screen_height // 2) - (box_height // 2) - vertical_offset
        
        self.fishing_region = (region_left, region_top, box_width, box_height)
        print(f"Focused detection region set: {self.fishing_region}")
        print(f"Monitoring {box_width}x{box_height} pixel area")
        print(f"Region center: X={region_left + box_width//2}, Y={region_top + box_height//2}")
        print(f"(Optimized for 2560x1440 - positioned left of center)")
        
    def detect_exclamation_marks(self):
        """Detect the exclamation marks pattern on screen - optimized for speed"""
        # Capture only the fishing region for faster processing
        screenshot = pyautogui.screenshot(region=self.fishing_region)
            
        # Convert to OpenCV format
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # Template matching with faster method
        result = cv2.matchTemplate(screenshot_gray, self.exclamation_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_val >= self.threshold
    
    def human_like_click(self):
        """Perform a right-click to catch fish"""
        # Right click to catch fish
        pyautogui.rightClick()
        print("Quick catch attempt!")
        
    def throw_fishing_rod(self):
        """Throw the fishing rod with random timing"""
        delay = random.uniform(self.throw_delay_min, self.throw_delay_max)
        time.sleep(delay)
        
        # Right click to throw
        pyautogui.rightClick()
        self.last_throw_time = time.time()  # Record throw time
        
        # Set a random timeout for this throw
        self.current_reset_timeout = random.uniform(self.reset_timeout_min, self.reset_timeout_max)
        
        print(f"Rod thrown after {delay:.2f}s delay (timeout: {self.current_reset_timeout:.1f}s)")
        
    def check_for_reset(self):
        """Check if we need to reset due to timeout"""
        current_time = time.time()
        time_since_throw = current_time - self.last_throw_time
        
        if time_since_throw > self.current_reset_timeout:
            print(f"No fish caught for {self.current_reset_timeout:.1f}s, resetting...")
            self.reset_fishing()
            return True
        return False
    
    def reset_fishing(self):
        """Reset the fishing state by throwing the rod again"""
        # Add a small random delay before reset
        reset_delay = random.uniform(0.5, 2.0)
        time.sleep(reset_delay)
        
        # Right click to reset (this will pull in the line if it's out, or throw if it's not)
        pyautogui.rightClick()
        time.sleep(0.5)  # Brief pause
        
        # Throw the rod again
        self.throw_fishing_rod()
        print("Fishing state reset successfully")
    
    def add_human_behavior(self):
        """Add random human-like behaviors"""
        if random.random() < self.break_chance:
            # Random short break
            break_time = random.uniform(2, 8)
            print(f"Taking a {break_time:.1f}s break...")
            time.sleep(break_time)
    
    def run(self):
        """Main fishing loop"""
        print("Auto-fisher starting in 5 seconds...")
        print("Press LEFT CLICK to stop the script at any time")
        print(f"Safety timer: Bot will auto-stop after {self.safety_stop_timer/3600:.1f} hour(s)")
        time.sleep(5)
        
        self.running = True
        self.start_time = time.time()  # Record start time for safety timer
        catches = 0
        resets = 0
        
        # Start mouse listener for panic button
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()
        
        # Initial throw
        self.throw_fishing_rod()
        
        while self.running and not self.stop_requested:
            try:
                # Check safety timer
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= self.safety_stop_timer:
                    print(f"\nSafety timer reached ({self.safety_stop_timer/3600:.1f} hour(s))!")
                    print("Stopping bot for safety...")
                    break
                
                # Check if we need to reset due to timeout
                if self.check_for_reset():
                    resets += 1
                    continue
                
                # Look for exclamation marks
                detected = self.detect_exclamation_marks()
                
                if detected:
                    # IMMEDIATE response for 0-1 second window
                    # Add tiny human-like reaction delay but keep it very short
                    reaction_delay = random.uniform(self.reaction_time_min, self.reaction_time_max)
                    time.sleep(reaction_delay)
                    
                    # Catch the fish immediately
                    self.human_like_click()
                    catches += 1
                    print(f"Fish caught! Total: {catches} (reaction: {reaction_delay:.3f}s)")
                    
                    # Throw again after delay
                    self.throw_fishing_rod()
                
                # Add occasional human behaviors during active fishing
                self.add_human_behavior()
                
                # Much smaller delay for faster detection loop
                time.sleep(self.detection_interval)
                
            except KeyboardInterrupt:
                print("Script interrupted by user")
                break
        
        # Clean up mouse listener
        if self.mouse_listener:
            self.mouse_listener.stop()
                
        self.running = False
        print(f"Auto-fisher stopped. Total catches: {catches}, Resets: {resets}")

def main():
    # Initialize with your exclamation marks image
    exclamation_image_path = "exclamation_marks.png"  # Update this path
    
    try:
        fisher = AutoFisher(exclamation_image_path)
        
        # Setup optimized detection region
        fisher.setup_center_left_region()
        
        # Start fishing
        fisher.run()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the exclamation_marks.png file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()