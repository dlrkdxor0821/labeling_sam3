"""Launch labelme without the opencv-python Qt-plugin conflict.

`opencv-python` (the non-headless wheel) hijacks QT_QPA_PLATFORM_PLUGIN_PATH on
import, pointing Qt at cv2's bundled (incompatible) xcb plugin — which makes
labelme crash with "Could not load the Qt platform plugin 'xcb'".

Fix: import cv2 FIRST (so it sets its broken path), then point Qt back at
PyQt5's own working xcb plugin before labelme creates its QApplication.

Usage (same args as labelme):
    python scripts/run_labelme.py datasets/bluebook/train/_needs_review/
"""
import os

import cv2  # noqa: F401 — import first so it applies its plugin-path override...
import PyQt5

# ...then override it back to PyQt5's working plugin dir.
_qt_platforms = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_platforms
os.environ.pop("QT_QPA_FONTDIR", None)  # drop cv2's font dir too (avoids warnings)

from labelme.__main__ import main  # noqa: E402 — must come after the env fix

if __name__ == "__main__":
    main()
