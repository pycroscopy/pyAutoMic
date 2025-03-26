# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>


import logging

def setup_logging(out_path: str = ".") -> None:
    """Setup logging to both console and file (log_time.txt)."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{out_path}/log_time.txt"),  # Logs to file
            logging.StreamHandler()  # Logs to console
        ]
    )

