# minesweeper_solver_debug.py
# Enhanced Minesweeper automation solver with debugging for click issues
# WARNING: Test in a safe environment first (use "dry_run" to avoid clicking).

import pyautogui
import time
import cv2
import numpy as np
import os
import threading
import keyboard
from collections import deque, defaultdict

pyautogui.PAUSE = 0.12
pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort instantly

# ---------- CONFIG ----------
DRY_RUN = False          # If True, the script will not click - only simulates moves
CLICK_DELAY = 0.06      # delay between clicks
BRIGHTNESS_THRESHOLD = 140  # threshold for covered/uncovered detection
TEMPLATE_THRESHOLD = 0.4   # template matching confidence (lowered for better detection)
PANIC_KEY = 'ctrl+shift+q'  # panic button combination
DEBUG_DETECTION = True   # show detailed number detection info
DEBUG_LOGIC = True       # NEW: show detailed logic debugging
ICONS_FOLDER = "icons"   # folder containing number templates
AUTO_START_GAME = True   # automatically click the green X to start
HIGH_DPI_MODE = True     # enable high DPI compatibility features
# ---------------------------

# Difficulty presets
DIFFICULTY_PRESETS = {
    '1': {'name': 'Easy', 'rows': 9, 'cols': 9},
    '2': {'name': 'Medium', 'rows': 16, 'cols': 16},
    '3': {'name': 'Hard', 'rows': 16, 'cols': 30},
    '4': {'name': 'Evil', 'rows': 20, 'cols': 30}
}

# Global flag for panic button
PANIC_STOP = False

def setup_panic_button():
    """Set up the panic button listener"""
    global PANIC_STOP
    
    def on_panic():
        global PANIC_STOP
        PANIC_STOP = True
        print("\n🚨 PANIC BUTTON PRESSED! Stopping solver immediately... 🚨")
    
    try:
        keyboard.add_hotkey(PANIC_KEY, on_panic)
        print(f"✅ Panic button set up: Press {PANIC_KEY} to stop the solver immediately")
    except Exception as e:
        print(f"⚠️ Could not set up panic button: {e}")
        print("You can still use Ctrl+C or move mouse to top-left corner to stop")

def check_panic():
    """Check if panic button was pressed"""
    global PANIC_STOP
    if PANIC_STOP:
        print("🛑 Solver stopped by panic button")
        return True
    return False

def select_difficulty():
    """Let user select difficulty preset"""
    print("\n🎯 Select Difficulty:")
    for key, preset in DIFFICULTY_PRESETS.items():
        print(f"{key}. {preset['name']} - {preset['rows']}x{preset['cols']}")
    
    while True:
        try:
            choice = input("\nEnter difficulty (1-4): ").strip()
            if choice in DIFFICULTY_PRESETS:
                preset = DIFFICULTY_PRESETS[choice]
                print(f"✅ Selected: {preset['name']} ({preset['rows']}x{preset['cols']})")
                return preset['rows'], preset['cols']
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n🛑 Cancelled by user")
            return None, None

def load_all_templates():
    """Load all template images from icons folder"""
    templates = {}
    
    if not os.path.exists(ICONS_FOLDER):
        print(f"❌ Icons folder '{ICONS_FOLDER}' not found!")
        print("Please create the folder and add your template images:")
        print("- number 1.png through number 5.png")
        print("- empty cell.png")
        print("- folded cell.png") 
        print("- flagged cell.png")
        print("- top left cell.png")
        print("- bottom right cell.png")
        print("- green x cell.png")
        return None
    
    print(f"📁 Loading templates from '{ICONS_FOLDER}' folder...")
    
    # Load number templates (1-5 only, since 6-8 are extremely rare)
    for i in range(1, 6):
        filename = f"number {i}.png"
        filepath = os.path.join(ICONS_FOLDER, filename)
        
        if os.path.exists(filepath):
            try:
                template = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[f'number_{i}'] = template
                    h, w = template.shape[:2]
                    print(f"✅ Loaded template {i}: {filename} ({w}x{h} pixels)")
                else:
                    print(f"⚠️ Could not read {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚠️ Missing: {filename}")
    
    # Load cell state templates
    cell_state_templates = [
        ('empty cell.png', 'empty'),
        ('folded cell.png', 'folded'),
        ('flagged cell.png', 'flagged')
    ]
    
    for filename, key in cell_state_templates:
        filepath = os.path.join(ICONS_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                template = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[key] = template
                    h, w = template.shape[:2]
                    print(f"✅ Loaded {key} template: {filename} ({w}x{h} pixels)")
                else:
                    print(f"⚠️ Could not read {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚠️ Missing: {filename}")
    
    # Load special templates for detection
    special_templates = ['top left cell.png', 'bottom right cell.png', 'green x cell.png']
    for template_name in special_templates:
        filepath = os.path.join(ICONS_FOLDER, template_name)
        if os.path.exists(filepath):
            print(f"✅ Found: {template_name}")
        else:
            print(f"⚠️ Missing: {template_name}")
    
    number_count = sum(1 for key in templates.keys() if key.startswith('number_'))
    state_count = sum(1 for key in ['empty', 'folded', 'flagged'] if key in templates)
    
    if number_count == 0:
        print("❌ No valid number templates found! Please add number images to icons folder.")
        return None
    
    print(f"🎯 Loaded {number_count} number templates and {state_count} cell state templates")
    return templates

def find_green_x_cell():
    """Find the green X starting cell using template matching"""
    green_x_template_path = os.path.join(ICONS_FOLDER, "green x cell.png")
    
    if not os.path.exists(green_x_template_path):
        print(f"⚠️ Missing: {green_x_template_path}")
        return None
    
    try:
        green_x_template = cv2.imread(green_x_template_path, cv2.IMREAD_COLOR)
        
        if green_x_template is None:
            print("❌ Could not load green X cell template")
            return None
            
        print("🔍 Searching for green X starting cell...")
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Find green X cell
        result = cv2.matchTemplate(screenshot_np, green_x_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.6:  # Lower threshold for detection
            print(f"❌ Could not find green X cell (confidence: {max_val:.3f})")
            return None
            
        # Calculate center of green X cell
        h, w = green_x_template.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        
        print(f"✅ Found green X cell at ({center_x}, {center_y}) - confidence: {max_val:.3f}")
        
        return (center_x, center_y)
        
    except Exception as e:
        print(f"❌ Error finding green X cell: {e}")
        return None

def click_green_x_to_start():
    """Find and click the green X to start the game"""
    green_x_pos = find_green_x_cell()
    
    if green_x_pos is None:
        print("⚠️ Could not find green X cell. Game might already be started or template missing.")
        return False
    
    x, y = green_x_pos
    print(f"🎯 Clicking green X at ({x}, {y}) to start game...")
    
    if DRY_RUN:
        print("(DRY RUN - not actually clicking)")
        return True
    
    try:
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click(button='left')
        time.sleep(0.5)  # Wait for game to initialize
        return True
    except Exception as e:
        print(f"❌ Error clicking green X: {e}")
        return False

def find_corner_cells():
    """Automatically find top-left and bottom-right cells using template matching"""
    
    # Load corner cell templates
    top_left_template_path = os.path.join(ICONS_FOLDER, "top left cell.png")
    bottom_right_template_path = os.path.join(ICONS_FOLDER, "bottom right cell.png")
    
    if not os.path.exists(top_left_template_path):
        print(f"❌ Missing: {top_left_template_path}")
        return None
    if not os.path.exists(bottom_right_template_path):
        print(f"❌ Missing: {bottom_right_template_path}")
        return None
    
    try:
        top_left_template = cv2.imread(top_left_template_path, cv2.IMREAD_COLOR)
        bottom_right_template = cv2.imread(bottom_right_template_path, cv2.IMREAD_COLOR)
        
        if top_left_template is None or bottom_right_template is None:
            print("❌ Could not load corner cell templates")
            return None
            
        print("🔍 Searching for corner cells...")
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Find top-left cell
        result_tl = cv2.matchTemplate(screenshot_np, top_left_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_tl, _, max_loc_tl = cv2.minMaxLoc(result_tl)
        
        if max_val_tl < 0.6:  # Lower threshold for corner detection
            print(f"❌ Could not find top-left cell (confidence: {max_val_tl:.3f})")
            return None
            
        # Calculate center of top-left cell
        tl_h, tl_w = top_left_template.shape[:2]
        tl_center_x = max_loc_tl[0] + tl_w // 2
        tl_center_y = max_loc_tl[1] + tl_h // 2
        
        # Find bottom-right cell
        result_br = cv2.matchTemplate(screenshot_np, bottom_right_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_br, _, max_loc_br = cv2.minMaxLoc(result_br)
        
        if max_val_br < 0.6:
            print(f"❌ Could not find bottom-right cell (confidence: {max_val_br:.3f})")
            return None
            
        # Calculate center of bottom-right cell  
        br_h, br_w = bottom_right_template.shape[:2]
        br_center_x = max_loc_br[0] + br_w // 2
        br_center_y = max_loc_br[1] + br_h // 2
        
        print(f"✅ Found top-left cell at ({tl_center_x}, {tl_center_y}) - confidence: {max_val_tl:.3f}")
        print(f"✅ Found bottom-right cell at ({br_center_x}, {br_center_y}) - confidence: {max_val_br:.3f}")
        
        return (tl_center_x, tl_center_y, br_center_x, br_center_y)
        
    except Exception as e:
        print(f"❌ Error finding corner cells: {e}")
        return None

def get_board_area_and_size(rows, cols):
    """Get board configuration using preselected difficulty"""
    print(f"🤖 Setting up board for {rows}x{cols} grid...")
    
    # Try automatic corner detection first
    corner_result = find_corner_cells()
    
    if corner_result is not None:
        x1, y1, x2, y2 = corner_result
        print("✅ Automatic corner detection successful!")
    else:
        print("⚠️ Automatic detection failed. Falling back to manual setup.")
        print("Move the mouse to the TOP-LEFT corner of the *first cell* (cell center) and press ENTER.")
        input("Ready? Press ENTER...")
        top_left_pos = pyautogui.position()
        print("Now move the mouse to the BOTTOM-RIGHT corner of the *last cell* (cell center) and press ENTER.")
        input("Ready? Press ENTER...")
        bottom_right_pos = pyautogui.position()
        x1, y1 = top_left_pos.x, top_left_pos.y
        x2, y2 = bottom_right_pos.x, bottom_right_pos.y
    
    print(f"Corner positions: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")
    
    # Calculate cell size based on center-to-center distance
    if cols > 1:
        cell_w = (x2 - x1) / (cols - 1)
    else:
        cell_w = 30  # fallback
    
    if rows > 1:
        cell_h = (y2 - y1) / (rows - 1)
    else:
        cell_h = 30  # fallback
    
    # Calculate board boundaries (extending half cell from centers)
    board_x1 = x1 - cell_w / 2
    board_y1 = y1 - cell_h / 2
    board_x2 = x2 + cell_w / 2
    board_y2 = y2 + cell_h / 2
    
    print(f"Calculated cell size: {cell_w:.1f} x {cell_h:.1f}")
    print(f"Board area: ({board_x1:.1f}, {board_y1:.1f}) to ({board_x2:.1f}, {board_y2:.1f})")
    return (board_x1, board_y1, board_x2, board_y2, rows, cols, cell_w, cell_h)

def cell_center(top_left_x, top_left_y, cell_w, cell_h, r, c):
    cx = top_left_x + cell_w / 2 + c * cell_w
    cy = top_left_y + cell_h / 2 + r * cell_h
    return int(cx), int(cy)

def is_flagged_color(pixel):
    """Check if pixel color indicates a flagged cell"""
    r, g, b = pixel
    # Red flag
    if r > 200 and g < 100 and b < 100:
        return True
    # Orange/yellow flag
    if r > 150 and g > 100 and b < 100:
        return True
    return False

def is_green_x_color(pixel):
    """Check if pixel color indicates a green X cell"""
    r, g, b = pixel
    # Green X (adjust thresholds based on your game's colors)
    if g > 150 and r < 120 and b < 120:
        return True
    return False

def detect_cell_content(screenshot_np, cell_rect, templates):
    """
    Use template matching to detect cell content
    Returns (content_type, value, confidence)
    content_type: 'number', 'empty', 'folded', 'flagged', 'green_x'
    value: number (1-5) for numbers, None for others
    """
    x, y, w, h = cell_rect
    
    # Extract cell region from screenshot
    cell_img = screenshot_np[y:y+h, x:x+w]
    
    if cell_img.size == 0:
        return 'unknown', None, 0
    
    best_match_type = None
    best_match_value = None
    best_confidence = 0
    
    # Lower threshold for high DPI displays
    detection_threshold = max(0.25, TEMPLATE_THRESHOLD - 0.15)  # Even lower threshold
    
    # Check cell state templates first (higher priority)
    state_templates = [
        ('flagged', 'flagged'),
        ('folded', 'folded'), 
        ('empty', 'empty')
    ]
    
    for template_key, content_type in state_templates:
        if template_key not in templates:
            continue
            
        template = templates[template_key]
        if template.shape[0] > h or template.shape[1] > w:
            continue
            
        # Try multiple scales for high DPI compatibility
        scales = [1.0, 0.8, 1.2, 0.6, 1.5, 0.5]  # Added more scales
        for scale in scales:
            if scale != 1.0:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                if scaled_template.shape[0] > h or scaled_template.shape[1] > w:
                    continue
            else:
                scaled_template = template
                
            result = cv2.matchTemplate(cell_img, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence and max_val > detection_threshold:
                best_confidence = max_val
                best_match_type = content_type
                best_match_value = None
    
    # Check number templates if no state template matched strongly
    if best_confidence < TEMPLATE_THRESHOLD * 0.8:  # Lower requirement for numbers
        for template_key, template in templates.items():
            if not template_key.startswith('number_'):
                continue
                
            if template.shape[0] > h or template.shape[1] > w:
                continue
                
            # Try multiple scales for high DPI compatibility
            scales = [1.0, 0.8, 1.2, 0.6, 1.5, 0.5]  # Added more scales
            for scale in scales:
                if scale != 1.0:
                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    if scaled_template.shape[0] > h or scaled_template.shape[1] > w:
                        continue
                else:
                    scaled_template = template
                    
                result = cv2.matchTemplate(cell_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence and max_val > detection_threshold:
                    best_confidence = max_val
                    best_match_type = 'number'
                    # Extract number from key (e.g., 'number_3' -> 3)
                    best_match_value = int(template_key.split('_')[1])
    
    return best_match_type, best_match_value, best_confidence

def read_board(top_left_x, top_left_y, rows, cols, cell_w, cell_h, templates):
    """
    Returns a matrix board with:
      -3 = green X (starting cell)
      -2 = flagged
      -1 = unknown/covered (folded)
       0 = revealed empty cell
       1-5 = revealed numbers
    """
    board = [[-1 for _ in range(cols)] for __ in range(rows)]
    
    # Take screenshot and convert to OpenCV format
    screenshot = pyautogui.screenshot()
    screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    if DEBUG_DETECTION:
        print("🔍 Starting board scan with enhanced template matching...")
        print(f"🖥️  High DPI mode: {HIGH_DPI_MODE}, Template threshold: {TEMPLATE_THRESHOLD}")
    
    detection_stats = {'template_success': 0, 'pixel_fallback': 0, 'total': 0}
    
    for r in range(rows):
        for c in range(cols):
            if check_panic():
                return board
                
            detection_stats['total'] += 1
            cx, cy = cell_center(top_left_x, top_left_y, cell_w, cell_h, r, c)
            
            # Define cell boundaries with some padding for high DPI
            padding = 3 if HIGH_DPI_MODE else 1  # Increased padding
            cell_left = int(cx - cell_w // 2 - padding)
            cell_top = int(cy - cell_h // 2 - padding)
            cell_right = int(cx + cell_w // 2 + padding)
            cell_bottom = int(cy + cell_h // 2 + padding)
            cell_width = cell_right - cell_left
            cell_height = cell_bottom - cell_top
            
            # Sample center pixel for fallback analysis
            try:
                center_pixel = screenshot.getpixel((cx, cy))
            except Exception:
                center_pixel = (128, 128, 128)
            
            # Check for green X first using pixel color (since it's special)
            if is_green_x_color(center_pixel):
                board[r][c] = -3
                detection_stats['pixel_fallback'] += 1
                if DEBUG_DETECTION:
                    print(f"  Cell ({r},{c}): GREEN X (start) - pixel color")
                continue
            
            # Use template matching for all other cell types
            cell_rect = (cell_left, cell_top, cell_width, cell_height)
            content_type, value, confidence = detect_cell_content(screenshot_np, cell_rect, templates)
            
            template_detected = False
            
            if content_type == 'flagged' and confidence > 0.25:  # Lower thresholds
                board[r][c] = -2
                template_detected = True
                if DEBUG_DETECTION:
                    print(f"  Cell ({r},{c}): FLAGGED (confidence: {confidence:.3f})")
                    
            elif content_type == 'folded' and confidence > 0.25:
                board[r][c] = -1  # Covered/unknown
                template_detected = True
                if DEBUG_DETECTION:
                    print(f"  Cell ({r},{c}): FOLDED (confidence: {confidence:.3f})")
                    
            elif content_type == 'empty' and confidence > 0.25:
                board[r][c] = 0  # Empty revealed
                template_detected = True
                if DEBUG_DETECTION:
                    print(f"  Cell ({r},{c}): EMPTY (confidence: {confidence:.3f})")
                    
            elif content_type == 'number' and value is not None and confidence > 0.25:
                board[r][c] = value
                template_detected = True
                if DEBUG_DETECTION:
                    print(f"  Cell ({r},{c}): NUMBER {value} (confidence: {confidence:.3f})")
            
            if template_detected:
                detection_stats['template_success'] += 1
            else:
                # Enhanced fallback to pixel-based detection
                detection_stats['pixel_fallback'] += 1
                center_brightness = sum(center_pixel) / 3
                
                if is_flagged_color(center_pixel):
                    board[r][c] = -2
                    if DEBUG_DETECTION:
                        print(f"  Cell ({r},{c}): FLAGGED (fallback pixel)")
                elif center_brightness < BRIGHTNESS_THRESHOLD - 30:  # More sensitive to covered cells
                    board[r][c] = -1  # Covered
                    if DEBUG_DETECTION:
                        print(f"  Cell ({r},{c}): COVERED (fallback pixel, brightness: {center_brightness:.1f})")
                elif center_brightness > BRIGHTNESS_THRESHOLD + 20:  # More sensitive to empty cells
                    board[r][c] = 0  # Empty revealed
                    if DEBUG_DETECTION:
                        print(f"  Cell ({r},{c}): EMPTY (fallback pixel, brightness: {center_brightness:.1f})")
                else:
                    board[r][c] = -1  # Default to covered
                    if DEBUG_DETECTION:
                        print(f"  Cell ({r},{c}): UNKNOWN (fallback default, brightness: {center_brightness:.1f})")
    
    template_success_rate = (detection_stats['template_success'] / detection_stats['total']) * 100
    print(f"✅ Board scan complete - Template success: {template_success_rate:.1f}% ({detection_stats['template_success']}/{detection_stats['total']})")
    
    if template_success_rate < 50:
        print("⚠️  Low template detection rate. Consider:")
        print("   - Lowering TEMPLATE_THRESHOLD further (try 0.25)")
        print("   - Checking template image quality and scaling")
        print("   - Ensuring game window is not scaled or zoomed")
    
    return board

def neighbors(rows, cols, r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc

def click_cell(top_left_x, top_left_y, cell_w, cell_h, r, c, button='left'):
    if check_panic():
        return False
        
    cx, cy = cell_center(top_left_x, top_left_y, cell_w, cell_h, r, c)
    print(f"Clicking {button} at cell ({r},{c}) -> screen ({cx},{cy})")
    
    if DRY_RUN:
        return True
    
    try:
        pyautogui.moveTo(cx, cy, duration=0.05)
        if button == 'left':
            pyautogui.click(button='left')
        else:
            pyautogui.click(button='right')
        time.sleep(CLICK_DELAY)
        return True
    except Exception as e:
        print(f"Error clicking cell: {e}")
        return False

def apply_deterministic(board, rows, cols):
    """
    Apply deterministic Minesweeper rules for No Guessing Mode:
      - if number == flagged count => open all unknown neighbors (all mines found)
      - if number == flagged count + unknown count => flag all unknowns (all remaining must be mines)
    Returns list of actions: (type, r, c) where type in {'click','flag'}
    """
    actions = []
    
    if DEBUG_LOGIC:
        print("🧠 Starting deterministic logic analysis...")
    
    for r in range(rows):
        for c in range(cols):
            val = board[r][c]
            if not (isinstance(val, int) and 1 <= val <= 5):
                continue
            
            unknowns = []
            flags = 0
            revealed = 0
            
            # Analyze all neighbors
            for nr, nc in neighbors(rows, cols, r, c):
                neighbor_val = board[nr][nc]
                if neighbor_val == -1:  # unknown/covered (not green X)
                    unknowns.append((nr, nc))
                elif neighbor_val == -2:  # flagged
                    flags += 1
                elif neighbor_val >= 0:  # revealed (empty or number)
                    revealed += 1
                # -3 (green X) is treated as safe/revealed for logic purposes
                elif neighbor_val == -3:
                    revealed += 1
            
            if len(unknowns) == 0:
                if DEBUG_LOGIC:
                    print(f"  Cell ({r},{c}) value={val}: No unknowns, skipping")
                continue  # no unknowns to work with
            
            remaining_mines = val - flags
            
            # Safety check
            if remaining_mines < 0:
                print(f"⚠️ WARNING: Cell ({r},{c}) has value {val} but {flags} flags - something is wrong!")
                continue
            
            if DEBUG_LOGIC:
                print(f"  Cell ({r},{c}) value={val}: {flags} flags, {len(unknowns)} unknowns, {revealed} revealed, need {remaining_mines} more mines")
            
            if remaining_mines == 0:
                # Rule 1: All mines already found - click all remaining unknowns
                print(f"🎯 LOGIC: Cell ({r},{c}) value={val}, flags={flags} -> clicking {len(unknowns)} unknowns (all mines found)")
                for (nr, nc) in unknowns:
                    actions.append(('click', nr, nc))
                    if DEBUG_LOGIC:
                        print(f"    -> Will CLICK ({nr},{nc}) - safe cell")
                    
            elif remaining_mines == len(unknowns):
                # Rule 2: All remaining unknowns must be mines
                print(f"🚩 LOGIC: Cell ({r},{c}) value={val}, flags={flags}, unknowns={len(unknowns)} -> flagging all unknowns")
                for (nr, nc) in unknowns:
                    actions.append(('flag', nr, nc))
                    if DEBUG_LOGIC:
                        print(f"    -> Will FLAG ({nr},{nc}) - must be mine")
            else:
                if DEBUG_LOGIC:
                    print(f"    -> No clear action (need {remaining_mines} mines from {len(unknowns)} unknowns)")
    
    # Remove duplicates, prioritizing flags over clicks for safety
    unique = {}
    for act in actions:
        typ, r, c = act
        key = (r, c)
        if key not in unique or typ == 'flag':
            unique[key] = typ
    
    final = [(typ, r, c) for (r, c), typ in unique.items()]
    
    if final:
        click_count = sum(1 for typ, r, c in final if typ == 'click')
        flag_count = sum(1 for typ, r, c in final if typ == 'flag')
        print(f"📋 Planned actions: {len(final)} total ({click_count} clicks, {flag_count} flags)")
        for typ, r, c in final:
            action_symbol = "🎯" if typ == 'click' else "🚩"
            print(f"  {action_symbol} {typ.upper()} at ({r},{c})")
    else:
        if DEBUG_LOGIC:
            print("📋 No deterministic actions found")
    
    return final

def find_safe_start_moves(board, rows, cols):
    """
    For No Guessing Mode: Find the green X cell and click it if available
    Returns list of safe starting moves
    """
    actions = []
    
    # Look for green X cell
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == -3:  # Green X
                print(f"🟢 Found green X at ({r},{c}) - this is the safe starting move")
                actions.append(('click', r, c))
                return actions
    
    # Also look for any obvious safe first moves (corners in easy mode, etc.)
    covered_cells = []
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == -1:  # Covered cell
                covered_cells.append((r, c))
    
    if covered_cells and not actions:
        # In No Guessing Mode, we should only click the green X or cells we're certain about
        print("⚠️ No green X found, but this is No Guessing Mode - cannot make random moves")
    
    return actions

def print_board(board):
    print("\n📋 Current board state:")
    for i, row in enumerate(board):
        line = []
        for j, cell in enumerate(row):
            if cell == -3:
                line.append('🟢')  # Green X (start) - more visible
            elif cell == -2:
                line.append('🚩')  # flagged
            elif cell == -1:
                line.append('❓')  # unknown
            elif cell == 0:
                line.append('⬜')  # empty revealed
            else:
                line.append(str(cell))  # number
        print(f"{i:2d}: " + ' '.join(f'{cell:>2}' for cell in line))
    print()
    
    # Show statistics
    covered = sum(row.count(-1) for row in board)
    flagged = sum(row.count(-2) for row in board) 
    green_x = sum(row.count(-3) for row in board)
    empty = sum(row.count(0) for row in board)
    numbers = sum(sum(1 for cell in row if isinstance(cell, int) and 1 <= cell <= 5) for row in board)
    total_cells = len(board) * len(board[0])
    
    print(f"📊 Stats: {covered} covered, {empty} empty, {numbers} numbers, {flagged} flagged, {green_x} green X ({total_cells} total)")
    
    # Additional debugging info
    if DEBUG_LOGIC:
        revealed_cells = empty + numbers
        if revealed_cells > 0:
            print(f"🔍 Debug: {revealed_cells} cells revealed, looking for deterministic moves...")
    print()

def main_loop(top_left_x, top_left_y, rows, cols, cell_w, cell_h, templates):
    print(f"🎮 Starting No Guessing Mode solver. Panic button: {PANIC_KEY}")
    print("You can also move mouse to top-left corner to abort (pyautogui FAILSAFE).")
    print(f"🐛 Debug mode: Detection={DEBUG_DETECTION}, Logic={DEBUG_LOGIC}")
    
    iteration = 0
    last_board_state = None
    stuck_count = 0
    max_stuck_iterations = 3  # Reduced for No Guessing Mode
    game_started = False
    
    while True:
        if check_panic():
            break
            
        iteration += 1
        print(f"\n{'='*20} Iteration {iteration} {'='*20}")
        
        try:
            board = read_board(top_left_x, top_left_y, rows, cols, cell_w, cell_h, templates)
            print_board(board)
            
            # Check if we need to start the game
            if not game_started:
                start_moves = find_safe_start_moves(board, rows, cols)
                if start_moves:
                    print("🎯 Starting game with safe move...")
                    typ, r, c = start_moves[0]
                    if click_cell(top_left_x, top_left_y, cell_w, cell_h, r, c, button='left'):
                        game_started = True
                        print("✅ Game started! Waiting for board to update...")
                        time.sleep(0.5)  # Give time for game to update
                        continue
                else:
                    # Auto-click green X if found via template matching
                    if AUTO_START_GAME:
                        print("🔍 Looking for green X to start game...")
                        if click_green_x_to_start():
                            game_started = True
                            print("✅ Game started via green X! Waiting for board to update...")
                            time.sleep(0.8)  # Give more time for game to update
                            continue
                        else:
                            print("⚠️ Could not find green X and no safe moves available")
                            print("Please manually start the game or check template images")
                            break
            
            # Check if board state has changed
            if last_board_state is not None and board == last_board_state:
                stuck_count += 1
                print(f"⚠️ Board unchanged for {stuck_count} iterations")
                if stuck_count >= max_stuck_iterations:
                    print(f"🏁 Board hasn't changed in {max_stuck_iterations} iterations.")
                    print("Possible causes:")
                    print("  - Templates not matching properly (check template images)")
                    print("  - Game window scaled/zoomed (try adjusting)")
                    print("  - Puzzle solved or needs manual intervention")
                    print("  - Clicking wrong locations (check cell detection)")
                    break
            else:
                stuck_count = 0
            
            last_board_state = [row[:] for row in board]  # deep copy
            
            # Apply deterministic logic (perfect for No Guessing Mode)
            actions = apply_deterministic(board, rows, cols)
            
            if actions:
                print(f"🚀 Executing {len(actions)} deterministic moves:")
                
                # Execute flags first, then clicks
                actions.sort(key=lambda x: 0 if x[0] == 'flag' else 1)
                
                for typ, r, c in actions:
                    if check_panic():
                        break
                        
                    # Double-check cell is still valid for action
                    current_cell = board[r][c]
                    if typ == 'click' and current_cell not in [-1, -3]:  # Can click unknown or green X
                        print(f"  ⚠️ Skipping {typ} at ({r},{c}) - cell no longer valid (current: {current_cell})")
                        continue
                    elif typ == 'flag' and current_cell != -1:  # Can only flag unknown
                        print(f"  ⚠️ Skipping {typ} at ({r},{c}) - cell no longer valid (current: {current_cell})")
                        continue
                    
                    action_symbol = "🎯" if typ == 'click' else "🚩"
                    print(f"  {action_symbol} {typ.capitalize()} cell ({r},{c})")
                    button = 'right' if typ == 'flag' else 'left'
                    
                    if not click_cell(top_left_x, top_left_y, cell_w, cell_h, r, c, button=button):
                        print("❌ Click failed, stopping")
                        return
                    
                    time.sleep(0.08)
                
                print("⏳ Waiting for game to update...")
                time.sleep(0.3)
                continue
            
            # No deterministic moves available in No Guessing Mode
            print("🎯 No deterministic moves available.")
            print("In No Guessing Mode, this means the puzzle should be solved!")
            
            # Count remaining cells
            covered = sum(row.count(-1) for row in board)
            green_x = sum(row.count(-3) for row in board)
            flagged = sum(row.count(-2) for row in board)
            
            if covered == 0 and green_x == 0:
                print("✅ All cells processed - game complete!")
            elif covered > 0:
                print(f"❓ {covered} cells remain covered, {flagged} flagged")
                print("This might indicate:")
                print("  - Template detection issues (cells not recognized properly)")
                print("  - The puzzle requires guessing (not truly No Guessing Mode)")
                print("  - Templates need adjustment for better accuracy")
            else:
                print(f"🟢 Only green X remains ({green_x} cells)")
            break
            
        except pyautogui.FailSafeException:
            print("🛑 FailSafe triggered - mouse moved to corner")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("🏁 No Guessing Mode solver finished.")

def run():
    print("🎯 Enhanced Minesweeper Solver - Debug Version")
    print("=" * 55)
    print(f"DRY RUN MODE: {'ON' if DRY_RUN else 'OFF'}")
    print(f"AUTO START: {'ON' if AUTO_START_GAME else 'OFF'}")
    print(f"DEBUG DETECTION: {'ON' if DEBUG_DETECTION else 'OFF'}")
    print(f"DEBUG LOGIC: {'ON' if DEBUG_LOGIC else 'OFF'}")
    if DRY_RUN:
        print("⚠️  Script will simulate moves without actually clicking")
    print()
    
    # Set up panic button
    setup_panic_button()
    
    # Select difficulty
    rows, cols = select_difficulty()
    if rows is None:
        return
    
    # Load templates
    templates = load_all_templates()
    if templates is None:
        print("❌ Cannot continue without templates. Please:")
        print("1. Create an 'icons' folder next to this script")
        print("2. Add PNG images: 'number 1.png' through 'number 5.png'")
        print("3. Add 'empty cell.png', 'folded cell.png', 'flagged cell.png'")
        print("4. Add 'top left cell.png', 'bottom right cell.png'")
        print("5. Add 'green x cell.png' for the starting position")
        print("6. Make sure the images show the elements clearly")
        return
    
    print("Make sure:")
    print("- Minesweeper game is visible and in No Guessing Mode")
    print("- Game board is not obstructed")
    print("- This console window stays accessible")
    print("- The green X starting position is visible (if game not started)")
    print("- Game window is not scaled or zoomed")
    print()
    
    try:
        # Get board configuration using preset
        x1, y1, x2, y2, rows, cols, cell_w, cell_h = get_board_area_and_size(rows, cols)
        
        print(f"\n🎯 Using No Guessing Mode with {len(templates)} templates")
        print(f"Template threshold: {TEMPLATE_THRESHOLD}")
        print(f"Board: {rows}x{cols}")
        print(f"Cell size: {cell_w:.1f}x{cell_h:.1f}")
        
        if not check_panic():
            input("\n✅ Ready to start solver? Press ENTER to begin (or use panic button to abort)...")
            
        if not check_panic():
            main_loop(x1, y1, rows, cols, cell_w, cell_h, templates)
            
    except pyautogui.FailSafeException:
        print("🛑 FailSafe triggered - script aborted by mouse position")
    except KeyboardInterrupt:
        print("🛑 Keyboard interrupt - exiting")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            keyboard.unhook_all_hotkeys()
        except:
            pass
        print("🏁 Solver terminated")

if __name__ == "__main__":
    # Check if OpenCV is available
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV not found! Please install it:")
        print("pip install opencv-python")
        exit(1)
    
    run()