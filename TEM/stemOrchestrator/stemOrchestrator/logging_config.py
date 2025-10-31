# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>


import logging
import sys


def setup_logging(out_path: str = ".") -> None:
    """Setup logging to both console and file (log_time.txt)."""

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{out_path}/log_time.txt"),
            logging.StreamHandler(sys.stdout),
        ],
    )
