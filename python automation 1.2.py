import pyautogui
import keyboard
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import json
import logging
from datetime import datetime
from pyautogui import ImageNotFoundException
import shutil
from tkinterdnd2 import DND_FILES, TkinterDnD

class ImageCache:
    """Handles caching of located images to improve performance"""
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30  # Cache expires after 30 seconds
    
    def get_cached_location(self, png_path):
        """Get cached location if it exists and hasn't expired"""
        if png_path in self.cache:
            location, timestamp = self.cache[png_path]
            if time.time() - timestamp < self.cache_timeout:
                return location
            else:
                # Remove expired cache entry
                del self.cache[png_path]
        return None
    
    def cache_location(self, png_path, location):
        """Cache the location of an image"""
        self.cache[png_path] = (location, time.time())
    
    def clear_cache(self):
        """Clear all cached locations"""
        self.cache.clear()

class ScreenCaptureWindow:
    """Window for capturing screen regions to create/update image assets"""
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        self.root = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        
    def start_capture(self):
        """Start the screen capture process"""
        # Take a screenshot first
        self.screenshot = pyautogui.screenshot()
        
        # Create fullscreen window
        self.root = tk.Toplevel(self.parent)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)
        self.root.configure(background='black')
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add instructions
        instructions = "Click and drag to select the area to capture. Press ESC to cancel."
        self.canvas.create_text(
            self.root.winfo_screenwidth()//2, 50,
            text=instructions, fill='white', font=('Arial', 16)
        )
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_selection)
        self.canvas.bind('<B1-Motion>', self.update_selection)
        self.canvas.bind('<ButtonRelease-1>', self.finish_selection)
        self.root.bind('<Escape>', self.cancel_capture)
        
        # Focus the window
        self.root.focus_force()
        self.canvas.focus_set()
    
    def start_selection(self, event):
        """Start drawing selection rectangle"""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        # Delete previous rectangle
        if self.rect:
            self.canvas.delete(self.rect)
    
    def update_selection(self, event):
        """Update selection rectangle as user drags"""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        # Delete previous rectangle
        if self.rect:
            self.canvas.delete(self.rect)
        
        # Draw new rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, cur_x, cur_y,
            outline='red', width=2
        )
    
    def finish_selection(self, event):
        """Finish selection and capture the area"""
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Calculate coordinates (ensure positive width/height)
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        
        # Close the capture window
        self.root.destroy()
        
        # Crop the screenshot to the selected area
        cropped_image = self.screenshot.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # Call the callback with the cropped image
        self.callback(cropped_image)
    
    def cancel_capture(self, event):
        """Cancel the capture process"""
        self.root.destroy()

class AutomationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("App Automation Tool")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize cache and logging
        self.image_cache = ImageCache()
        self.setup_logging()
        
        # Ensure icons directory exists
        if not os.path.exists("icons"):
            os.makedirs("icons")
        
        # Load app configuration
        self.load_app_config()
        
        # Create GUI elements
        self.create_widgets()
        
        # Setup drag and drop
        self.setup_drag_and_drop()
        
        # Status tracking
        self.is_running = False
    
    def setup_logging(self):
        """Setup logging to display in GUI"""
        self.log_messages = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    def load_app_config(self):
        """Load app configuration from file or use default"""
        config_file = "app_config.json"
        
        # Default configuration
        default_apps = {
            'google': "icons/google.png",
            'whatsapp': "icons/whatsapp.png",
            'youtube': "icons/youtube.png",
            'education gov il': "icons/education gov il.png",
            'education gov il2': "icons/education gov il2.png",
            'education gov il3': "icons/education gov il3.png",
            'education gov il4': "icons/education gov il4.png",
            'education gov il5': "icons/education gov il5.png",
            'education gov il6': "icons/education gov il6.png",
            'education gov il7': "icons/education gov il7.png",
            'education gov il8': "icons/education gov il8.png",
            'new tab': "icons/new tab.png",
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.apps = json.load(f)
            else:
                self.apps = default_apps
                self.save_app_config()
        except Exception as e:
            self.apps = default_apps
            self.log_message(f"Error loading config: {e}, using defaults")
    
    def save_app_config(self):
        """Save current app configuration to file"""
        try:
            with open("app_config.json", 'w') as f:
                json.dump(self.apps, f, indent=2)
            self.log_message("Configuration saved successfully")
        except Exception as e:
            self.log_message(f"Error saving config: {e}")
    
    def setup_drag_and_drop(self):
        """Setup drag and drop functionality for the entire window"""
        # Enable drag and drop for the main window
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<DropEnter>>', self.on_drop_enter)
        self.root.dnd_bind('<<DropPosition>>', self.on_drop_position)
        self.root.dnd_bind('<<DropLeave>>', self.on_drop_leave)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop_enter(self, event):
        """Handle when files enter the drop zone"""
        self.root.configure(bg='lightblue')
        return event.action
    
    def on_drop_position(self, event):
        """Handle drag position updates"""
        return event.action
    
    def on_drop_leave(self, event):
        """Handle when files leave the drop zone"""
        self.root.configure(bg='SystemButtonFace')
        return event.action
    
    def on_drop(self, event):
        """Handle dropped files"""
        self.root.configure(bg='SystemButtonFace')
        files = event.data.split()
        
        for file_path in files:
            # Remove curly braces that sometimes appear
            file_path = file_path.strip('{}')
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.handle_dropped_image(file_path)
            else:
                self.log_message(f"Skipped non-image file: {os.path.basename(file_path)}")
        
        return event.action
    
    def handle_dropped_image(self, file_path):
        """Handle a dropped image file"""
        filename = os.path.basename(file_path)
        
        # Ask user for app name
        app_name = tk.simpledialog.askstring(
            "Add New App", 
            f"Enter a name for this app icon:\n{filename}",
            initialvalue=os.path.splitext(filename)[0]
        )
        
        if app_name:
            # Copy file to icons directory
            destination = os.path.join("icons", f"{app_name}.png")
            
            try:
                # Convert to PNG if necessary
                from PIL import Image
                with Image.open(file_path) as img:
                    img.save(destination, "PNG")
                
                # Add to configuration
                self.apps[app_name] = destination
                self.save_app_config()
                
                # Update dropdown
                self.app_dropdown['values'] = list(self.apps.keys())
                
                self.log_message(f"✅ Added new app: {app_name}")
                
            except Exception as e:
                self.log_message(f"❌ Error adding app {app_name}: {e}")
                messagebox.showerror("Error", f"Failed to add app: {e}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="App Automation Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Drag and drop area
        drop_frame = ttk.LabelFrame(main_frame, text="Drag & Drop Image Files Here", padding="10")
        drop_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        drop_label = ttk.Label(drop_frame, 
                              text="Drop PNG/JPG files here to add them as new automation targets",
                              font=('Arial', 10, 'italic'))
        drop_label.pack()
        
        # App selection
        ttk.Label(main_frame, text="Select App:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.app_var = tk.StringVar()
        self.app_dropdown = ttk.Combobox(main_frame, textvariable=self.app_var, 
                                        values=list(self.apps.keys()), 
                                        state="readonly", width=30)
        self.app_dropdown.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Update image button
        self.update_image_button = ttk.Button(main_frame, text="Update Image", 
                                            command=self.update_selected_app_image)
        self.update_image_button.grid(row=2, column=2, padx=(10, 0))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Open App button
        self.open_button = ttk.Button(button_frame, text="Open App", 
                                     command=self.open_app_threaded, style="Accent.TButton")
        self.open_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear Cache button
        self.clear_cache_button = ttk.Button(button_frame, text="Clear Cache", 
                                           command=self.clear_cache)
        self.clear_cache_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_automation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Settings button
        self.settings_button = ttk.Button(button_frame, text="Settings", 
                                        command=self.open_settings)
        self.settings_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Capture button
        self.capture_button = ttk.Button(button_frame, text="Capture Screen", 
                                       command=self.start_screen_capture)
        self.capture_button.pack(side=tk.LEFT)
        
        # Log display
        ttk.Label(main_frame, text="Activity Log:").grid(row=4, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=70)
        self.log_text.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Drop image files anywhere to add new apps")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def update_selected_app_image(self):
        """Update the image for the currently selected app"""
        selected_app = self.app_var.get()
        if not selected_app:
            messagebox.showwarning("Warning", "Please select an app first!")
            return
        
        self.log_message(f"Starting image update for: {selected_app}")
        self.start_screen_capture(selected_app)
    
    def start_screen_capture(self, app_name=None):
        """Start screen capture to create or update an app image"""
        if not app_name:
            app_name = tk.simpledialog.askstring(
                "New App", 
                "Enter a name for the new app:"
            )
            if not app_name:
                return
        
        self.log_message("Starting screen capture - select the area to capture")
        
        # Hide the main window temporarily
        self.root.withdraw()
        
        def on_capture_complete(cropped_image):
            # Show the main window again
            self.root.deiconify()
            
            # Save the captured image
            filename = f"{app_name}.png"
            filepath = os.path.join("icons", filename)
            
            try:
                cropped_image.save(filepath, "PNG")
                
                # Update configuration
                self.apps[app_name] = filepath
                self.save_app_config()
                
                # Update dropdown
                self.app_dropdown['values'] = list(self.apps.keys())
                
                # Clear cache for this image
                if filepath in self.image_cache.cache:
                    del self.image_cache.cache[filepath]
                
                self.log_message(f"✅ Image updated/created for: {app_name}")
                messagebox.showinfo("Success", f"Image for '{app_name}' has been updated!")
                
            except Exception as e:
                self.log_message(f"❌ Error saving image for {app_name}: {e}")
                messagebox.showerror("Error", f"Failed to save image: {e}")
        
        # Start the capture window
        capture_window = ScreenCaptureWindow(self.root, on_capture_complete)
        capture_window.start_capture()
    
    def log_message(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, status):
        """Update status bar"""
        self.status_var.set(status)
        self.root.update_idletasks()
    
    def wait_and_click(self, png_path, timeout=10, confidence=0.9):
        """Wait for image and click it, using cache when possible"""
        self.log_message(f"Looking for: {os.path.basename(png_path)}")
        
        start_time = time.time()
        
        # Check cache first
        cached_location = self.image_cache.get_cached_location(png_path)
        if cached_location:
            self.log_message(f"Using cached location for {os.path.basename(png_path)}")
            try:
                x, y = pyautogui.center(cached_location)
                pyautogui.moveTo(x, y, duration=0.1)
                pyautogui.click()
                return True
            except:
                # Cache might be stale, clear it and continue with normal search
                self.image_cache.cache.pop(png_path, None)
        
        # Normal image search
        while time.time() - start_time < timeout and self.is_running:
            try:
                location = pyautogui.locateOnScreen(png_path, confidence=confidence)
                if location:
                    # Cache the found location
                    self.image_cache.cache_location(png_path, location)
                    
                    x, y = pyautogui.center(location)
                    pyautogui.moveTo(x, y, duration=0.1)
                    pyautogui.click()
                    self.log_message(f"Clicked: {os.path.basename(png_path)}")
                    return True
            except ImageNotFoundException:
                pass
            
            if not self.is_running:
                return False
                
            time.sleep(0.2)
        
        self.log_message(f"❌ {os.path.basename(png_path)} not found within {timeout} seconds")
        return False
    
    def open_app_logic(self, app_name):
        """Core app opening logic"""
        try:
            self.is_running = True
            self.log_message(f"🚀 Starting automation for: {app_name}")
            
            # Always click Google first
            if not self.wait_and_click(self.apps['google']):
                return False
            
            # Handle new tab
            try:
                location = pyautogui.locateOnScreen(self.apps['new tab'], confidence=0.9)
                if location:
                    x, y = pyautogui.center(location)
                    pyautogui.moveTo(x, y, duration=0.1)
                    pyautogui.click()
                    self.log_message("Clicked existing new tab button")
                else:
                    self.log_message("Creating new tab with Ctrl+T")
                    pyautogui.hotkey('ctrl', 't')
                    time.sleep(0.5)
            except ImageNotFoundException:
                self.log_message("Creating new tab with Ctrl+T")
                pyautogui.hotkey('ctrl', 't')
                time.sleep(0.5)
            
            if not self.is_running:
                return False
            
            # Focus search bar and search
            pyautogui.hotkey('ctrl', 'e')
            pyautogui.write(app_name)
            pyautogui.hotkey('enter')
            self.log_message(f"Searched for: {app_name}")
            
            time.sleep(1)  # Wait for search results
            
            if not self.is_running:
                return False
            
            # Click the requested app
            if app_name in self.apps:
                if not self.wait_and_click(self.apps[app_name]):
                    return False
            else:
                self.log_message(f"❌ {app_name} is not configured")
                return False
            
            # Special handling for education gov il
            if app_name == 'education gov il':
                sequence = ['education gov il2', 'education gov il3', 'education gov il4',
                           'education gov il5', 'education gov il6', 'education gov il7',
                           'education gov il8']
                
                for step in sequence:
                    if not self.is_running:
                        return False
                    if not self.wait_and_click(self.apps[step]):
                        self.log_message(f"⚠️ Failed to complete step: {step}")
                        # Continue anyway - some steps might be optional
            
            self.log_message(f"✅ Successfully opened: {app_name}")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error during automation: {str(e)}")
            return False
        finally:
            self.is_running = False
    
    def open_app_threaded(self):
        """Open app in a separate thread to prevent GUI freezing"""
        app_name = self.app_var.get()
        if not app_name:
            messagebox.showwarning("Warning", "Please select an app first!")
            return
        
        # Disable buttons during operation
        self.open_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Running automation...")
        
        def automation_thread():
            try:
                success = self.open_app_logic(app_name)
                
                # Re-enable buttons
                self.root.after(0, lambda: self.open_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                
                if success:
                    self.root.after(0, lambda: self.update_status("Automation completed successfully"))
                else:
                    self.root.after(0, lambda: self.update_status("Automation failed or was stopped"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Thread error: {str(e)}"))
                self.root.after(0, lambda: self.open_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.update_status("Error occurred"))
        
        threading.Thread(target=automation_thread, daemon=True).start()
    
    def stop_automation(self):
        """Stop the currently running automation"""
        self.is_running = False
        self.log_message("🛑 Stopping automation...")
        self.update_status("Stopping...")
    
    def clear_cache(self):
        """Clear the image cache"""
        self.image_cache.clear_cache()
        self.log_message("🗑️ Image cache cleared")
        messagebox.showinfo("Cache Cleared", "Image cache has been cleared successfully!")
    
    def open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        settings_window.resizable(True, True)
        
        # Settings content
        ttk.Label(settings_window, text="App Configuration", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # App list frame
        apps_frame = ttk.LabelFrame(settings_window, text="Configured Apps", padding="10")
        apps_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create treeview for app list
        tree_frame = ttk.Frame(apps_frame)
        tree_frame.pack(fill="both", expand=True)
        
        # Treeview
        tree = ttk.Treeview(tree_frame, columns=("path",), show="tree headings")
        tree.heading("#0", text="App Name")
        tree.heading("path", text="Image Path")
        tree.column("#0", width=200)
        tree.column("path", width=300)
        
        # Populate treeview
        for app_name, app_path in self.apps.items():
            exists = "✅" if os.path.exists(app_path) else "❌"
            tree.insert("", "end", text=f"{exists} {app_name}", values=(app_path,))
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons frame
        buttons_frame = ttk.Frame(apps_frame)
        buttons_frame.pack(fill="x", pady=(10, 0))
        
        def delete_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "Please select an app to delete.")
                return
            
            item = tree.item(selected[0])
            app_name = item["text"].replace("✅ ", "").replace("❌ ", "")
            
            if messagebox.askyesno("Confirm Delete", f"Delete '{app_name}' from configuration?"):
                if app_name in self.apps:
                    del self.apps[app_name]
                    self.save_app_config()
                    self.app_dropdown['values'] = list(self.apps.keys())
                    tree.delete(selected[0])
                    self.log_message(f"Deleted app: {app_name}")
        
        def refresh_list():
            # Clear and repopulate tree
            for item in tree.get_children():
                tree.delete(item)
            
            for app_name, app_path in self.apps.items():
                exists = "✅" if os.path.exists(app_path) else "❌"
                tree.insert("", "end", text=f"{exists} {app_name}", values=(app_path,))
        
        ttk.Button(buttons_frame, text="Delete Selected", command=delete_selected).pack(side="left", padx=(0, 10))
        ttk.Button(buttons_frame, text="Refresh", command=refresh_list).pack(side="left", padx=(0, 10))
        
        # Cache settings
        cache_frame = ttk.LabelFrame(settings_window, text="Cache Settings", padding="10")
        cache_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(cache_frame, text=f"Cache timeout: {self.image_cache.cache_timeout} seconds").pack()
        ttk.Label(cache_frame, text=f"Cached items: {len(self.image_cache.cache)}").pack()
        
        ttk.Button(cache_frame, text="Clear Cache", 
                  command=self.clear_cache).pack(pady=5)

def main():
    # Create TkinterDnD root instead of regular Tk
    root = TkinterDnD.Tk()
    
    # Check for required libraries
    try:
        from PIL import Image
        import tkinterdnd2
    except ImportError as e:
        messagebox.showerror("Missing Dependencies", 
                           f"Required library not found: {e}\n\n"
                           "Please install missing dependencies:\n"
                           "pip install Pillow tkinterdnd2")
        return
    
    app = AutomationGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")

if __name__ == "__main__":
    main()