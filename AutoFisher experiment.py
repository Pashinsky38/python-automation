import pyautogui
import cv2
import numpy as np
import time
import random
import keyboard

class AutoFisher:
    def __init__(self, exclamation_image_path):
        self.exclamation_template = cv2.imread(exclamation_image_path, 0)
        self.running = False
        self.last_catch_time = 0
        self.last_throw_time = 0
        self.fishing_region = None  # You can set this to focus on a specific screen area
        
        # Human-like timing parameters
        self.reaction_time_min = 0.03  # Very fast minimum reaction time
        self.reaction_time_max = 0.1   # Fast maximum reaction time
        self.throw_delay_min = 0.1     # Much faster min delay before throwing again
        self.throw_delay_max = 0.3     # Much faster max delay before throwing again
        
        # Reset mechanism
        self.reset_timeout = 12.0     # Reset after 10 seconds of no detection
        
        # Detection threshold (lower for faster detection)
        self.threshold = 0.58  # Slightly lower threshold for faster matching
        
        # Fast detection settings
        self.detection_interval = 0.01  # Check every 10ms for even faster detection
        
    def setup_fishing_area(self):
        """Let user select the fishing area on screen"""
        print("Move your mouse to the top-left corner of the fishing area and press 'q'")
        while not keyboard.is_pressed('q'):
            time.sleep(0.1)
        
        x1, y1 = pyautogui.position()
        
        print("Now move to bottom-right corner and press 'q'")
        while not keyboard.is_pressed('q'):
            time.sleep(0.1)
            
        x2, y2 = pyautogui.position()
        
        self.fishing_region = (x1, y1, x2-x1, y2-y1)
        print(f"Fishing area set: {self.fishing_region}")
    
    def setup_left_side_region(self):
        """Automatically set fishing region to left half of screen"""
        screen_width, screen_height = pyautogui.size()
        
        # Set region to left half of screen
        # You can adjust these values as needed
        left_width = screen_width // 2  # Left half
        
        self.fishing_region = (0, 0, left_width, screen_height)
        print(f"Left side fishing region set: {self.fishing_region}")
        print(f"Monitoring left {left_width}x{screen_height} pixels of screen")
        
    def detect_exclamation_marks(self):
        """Detect the exclamation marks pattern on screen - optimized for speed"""
        if self.fishing_region:
            # Capture only the fishing region for faster processing
            screenshot = pyautogui.screenshot(region=self.fishing_region)
        else:
            # Capture full screen
            screenshot = pyautogui.screenshot()
            
        # Convert to OpenCV format
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # Template matching with faster method
        result = cv2.matchTemplate(screenshot_gray, self.exclamation_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= self.threshold:
            # Found exclamation marks
            return True, max_loc
        return False, None
    
    def human_like_click(self):
        """Perform a right-click with minimal delay for fast catching"""
        # For 0-1 second windows, we need to be much faster
        # Still add tiny randomness but keep it quick
        current_x, current_y = pyautogui.position()
        offset_x = random.randint(-2, 2)  # Smaller offset for speed
        offset_y = random.randint(-2, 2)
        
        # Very quick movement
        pyautogui.moveTo(current_x + offset_x, current_y + offset_y, 
                        duration=random.uniform(0.01, 0.05))  # Much faster
        
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
        print(f"Rod thrown after {delay:.2f}s delay")
        
    def check_for_reset(self):
        """Check if we need to reset due to timeout"""
        current_time = time.time()
        time_since_throw = current_time - self.last_throw_time
        
        if time_since_throw > self.reset_timeout:
            print(f"No fish caught for {self.reset_timeout}s, resetting...")
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
        if random.random() < 0.1:  # 10% chance
            # Random small mouse movement
            current_x, current_y = pyautogui.position()
            new_x = current_x + random.randint(-50, 50)
            new_y = current_y + random.randint(-50, 50)
            pyautogui.moveTo(new_x, new_y, duration=random.uniform(0.5, 1.5))
            
        if random.random() < 0.05:  # 5% chance
            # Random short break
            break_time = random.uniform(2, 8)
            print(f"Taking a {break_time:.1f}s break...")
            time.sleep(break_time)
        """Add random human-like behaviors"""
        if random.random() < 0.1:  # 10% chance
            # Random small mouse movement
            current_x, current_y = pyautogui.position()
            new_x = current_x + random.randint(-50, 50)
            new_y = current_y + random.randint(-50, 50)
            pyautogui.moveTo(new_x, new_y, duration=random.uniform(0.5, 1.5))
            
        if random.random() < 0.05:  # 5% chance
            # Random short break
            break_time = random.uniform(2, 8)
            print(f"Taking a {break_time:.1f}s break...")
            time.sleep(break_time)
    
    def run(self):
        """Main fishing loop"""
        print("Auto-fisher starting in 5 seconds...")
        print("Press 'ESC' to stop the script")
        time.sleep(5)
        
        self.running = True
        catches = 0
        resets = 0
        
        # Initial throw
        self.throw_fishing_rod()
        
        while self.running:
            try:
                # Check for stop command
                if keyboard.is_pressed('esc'):
                    print("Stopping auto-fisher...")
                    break
                
                # Check if we need to reset due to timeout
                if self.check_for_reset():
                    resets += 1
                    continue
                
                # Look for exclamation marks
                detected, location = self.detect_exclamation_marks()
                
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
                    
                    # Record catch time
                    self.last_catch_time = time.time()
                
                # Reduce human behaviors frequency during active fishing
                if random.random() < 0.02:  # Reduced from 0.1 to 0.02 for speed
                    self.add_human_behavior()
                
                # Much smaller delay for faster detection loop
                time.sleep(self.detection_interval)
                
            except KeyboardInterrupt:
                print("Script interrupted by user")
                break
                
        self.running = False
        print(f"Auto-fisher stopped. Total catches: {catches}, Resets: {resets}")

def main():
    # Initialize with your exclamation marks image
    exclamation_image_path = "exclamation_marks.png"  # Update this path
    
    try:
        fisher = AutoFisher(exclamation_image_path)
        
        # Setup options
        print("Choose fishing region setup:")
        print("1. Left side of screen (automatic)")
        print("2. Custom region (manual selection)")
        print("3. Full screen")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            fisher.setup_left_side_region()
        elif choice == "2":
            fisher.setup_fishing_area()
        elif choice == "3":
            print("Using full screen detection")
        else:
            print("Invalid choice, using left side by default")
            fisher.setup_left_side_region()
        
        # Start fishing
        fisher.run()
        
    except FileNotFoundError:
        print(f"Error: Could not find exclamation marks image at {exclamation_image_path}")
        print("Please make sure the image file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()