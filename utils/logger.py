import logging
import os
import sys
import warnings
import traceback

line_seg = ''.join(['*'] * 10)
logger = logging.getLogger("transnet")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(levelname).1s %(asctime)s %(filename)s:%(lineno)-4d] %(message)s',
        datefmt='%m.%d/%H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, "run.log"), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Capture warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(stream_handler)
    warnings_logger.addHandler(file_handler)

    # Capture unhandled exceptions
    sys.excepthook = handle_exception

    return logger
