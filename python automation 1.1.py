import pyautogui
import keyboard
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import logging
from datetime import datetime
from pyautogui import ImageNotFoundException

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

class AutomationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("App Automation Tool")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Initialize cache and logging
        self.image_cache = ImageCache()
        self.setup_logging()
        
        # Load app configuration
        self.load_app_config()
        
        # Create GUI elements
        self.create_widgets()
        
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
        except Exception as e:
            self.log_message(f"Error saving config: {e}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="App Automation Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # App selection
        ttk.Label(main_frame, text="Select App:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.app_var = tk.StringVar()
        self.app_dropdown = ttk.Combobox(main_frame, textvariable=self.app_var, 
                                        values=list(self.apps.keys()), 
                                        state="readonly", width=30)
        self.app_dropdown.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
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
        self.settings_button.pack(side=tk.LEFT)
        
        # Log display
        ttk.Label(main_frame, text="Activity Log:").grid(row=3, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=70)
        self.log_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
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
        settings_window.geometry("500x400")
        settings_window.resizable(True, True)
        
        # Settings content
        ttk.Label(settings_window, text="App Configuration", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(settings_window, 
                               text="Edit app_config.json to modify app paths and add new apps.")
        instructions.pack(pady=5)
        
        # Cache settings
        cache_frame = ttk.LabelFrame(settings_window, text="Cache Settings", padding="10")
        cache_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(cache_frame, text=f"Cache timeout: {self.image_cache.cache_timeout} seconds").pack()
        ttk.Label(cache_frame, text=f"Cached items: {len(self.image_cache.cache)}").pack()
        
        ttk.Button(cache_frame, text="Clear Cache", 
                  command=self.clear_cache).pack(pady=5)

def main():
    root = tk.Tk()
    app = AutomationGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")

if __name__ == "__main__":
    main()