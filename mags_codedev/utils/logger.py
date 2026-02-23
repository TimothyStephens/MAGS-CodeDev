import logging
from rich.logging import RichHandler

def setup_logger():
    """Configures dual logging: plain-text to file, colored to terminal."""
    logger = logging.getLogger("mags_codedev")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if called multiple times
    if logger.handlers:
        return logger

    # 1. File Handler (Plain text, no colors)
    file_handler = logging.FileHandler("mags-codedev_workflow.log")
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)

    # 2. Console Handler (Rich formatting)
    console_handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()