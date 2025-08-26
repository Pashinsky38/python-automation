import pyautogui, keyboard, time, os
from pyautogui import ImageNotFoundException

# Map app names to their PNG file paths
apps = {
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

def wait_and_click(png_path, timeout=10, confidence=0.9):
    """
    Wait up to `timeout` seconds for an image to appear on screen, then click it.
    If not found, prints a message and returns False.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            location = pyautogui.locateOnScreen(png_path, confidence=confidence)
            if location:
                x, y = pyautogui.center(location)
                pyautogui.moveTo(x, y, duration=0.1)
                pyautogui.click()
                return True
        except ImageNotFoundException:
            # Just means not found in this frame, keep looping
            pass
        time.sleep(0.2)  # small delay before retrying
    print(f"Icon {png_path} not found within {timeout} seconds.")
    return False

def open_app(user_input):
    """Open an app by clicking its icon and handling special cases."""
    # Always click Google first
    wait_and_click(apps['google'])

    # Try to detect if a "new tab" is available
    try:
        location = pyautogui.locateOnScreen(apps['new tab'], confidence=0.9)
        if location:
            x, y = pyautogui.center(location)
            pyautogui.moveTo(x, y, duration=0.1)
            pyautogui.click()
        else:
            print("No existing 'new tab' button detected, creating a new one with Ctrl+T.")
            pyautogui.hotkey('ctrl', 't')
            time.sleep(0.5)  # small delay so the new tab loads
    except ImageNotFoundException:
        print("No existing 'new tab' button detected, creating a new one with Ctrl+T.")
        pyautogui.hotkey('ctrl', 't')
        time.sleep(0.5)

    # Focus search bar and type the query
    pyautogui.hotkey('ctrl', 'e')
    pyautogui.write(user_input)
    pyautogui.hotkey('enter')

    # Then click the app the user requested
    if user_input in apps:
        wait_and_click(apps[user_input])
    else:
        print(f"{user_input} is not a recognized app.")
        return

    # Special handling for "education gov il"
    if user_input == 'education gov il':
        wait_and_click(apps['education gov il2'])
        wait_and_click(apps['education gov il3'])
        wait_and_click(apps['education gov il4'])
        wait_and_click(apps['education gov il5'])
        wait_and_click(apps['education gov il6'])
        wait_and_click(apps['education gov il7'])
        wait_and_click(apps['education gov il8'])


def main_project():
    user_input = input("Enter the name of the app to open (e.g., 'google', 'whatsapp'): ").strip().lower()
    if user_input not in apps:
        print(f"{user_input} is not a valid input, please try again.\n")
        return
    open_app(user_input)

while True:
    main_project()
