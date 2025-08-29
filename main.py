import sys, os

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else base_path
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from ui.welcome_window import WelcomeWindow
from ui.visu_window import VisuWindow  
from ui.result_window import ResultWindow     

def resource_path(relative_path):
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    icon  = resource_path(os.path.join("media", "logoClair.ico"))
    image = resource_path(os.path.join("media", "logoVclair.png"))
    WelcomeWindow.icon_path = icon
    WelcomeWindow.image_path = image

    app = WelcomeWindow()
    app.mainloop()
