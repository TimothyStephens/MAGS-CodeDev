import logging

def setup_logger():
    """Configures logging: plain-text to file only."""
    logger = logging.getLogger("mags_codedev")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate logs if called multiple times
    if logger.handlers:
        return logger

    # 1. File Handler (Plain text, no colors)
    file_handler = logging.FileHandler("mags-codedev_workflow.log", encoding="utf-8")
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()