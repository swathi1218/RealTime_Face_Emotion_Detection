"""
EmotionLens — Entry Point.

Configures logging, creates the Qt application, and launches the main window.
"""

import logging
import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensure the src/ directory is on sys.path so all internal packages resolve
# correctly whether the app is launched as a script or via the CLI entry point.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def _configure_logging() -> None:
    """Set up root logger with a sensible default format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Initialise and run the EmotionLens application."""
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting EmotionLens…")

    # Enable high-DPI scaling on Qt6 (no-op on Qt5; kept for portability)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("EmotionLens")
    app.setOrganizationName("EmotionLens")
    app.setStyle("Fusion")  

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
