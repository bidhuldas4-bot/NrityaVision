"""NrityaVision AI - entry point.   Run:  python main.py"""
import logging
import sys


def check_dependencies():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-contrib-python"), ("mediapipe", "mediapipe"),
                     ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("PIL", "Pillow"),
                     ("reportlab", "reportlab"), ("tkinter", "tkinter (reinstall Python with 'tcl/tk' enabled)")):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{pkg}  ({exc})")
    if missing:
        print("Missing or broken packages:\n  " + "\n  ".join(missing))
        print("\nFix:  pip install -r requirements.txt   (use Python 3.10-3.12, 64-bit)")
        sys.exit(1)


def main():
    check_dependencies()
    from nrityavision import config as C
    logging.basicConfig(filename=C.LOG_DIR / "app.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from nrityavision.ui.app import App
    App().mainloop()


if __name__ == "__main__":
    main()
