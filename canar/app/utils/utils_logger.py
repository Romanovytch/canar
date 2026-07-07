import logging
import os


#configuration du logger
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "canar.log")

if not os.path.exists(LOG_DIR) :
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)

sys_logger = logging.getLogger("Canar")
