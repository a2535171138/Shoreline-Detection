import os
import gdown
import logging
import threading
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/app/backend')

MODEL_PATHS = {
    'General': os.path.join(BASE_PATH, "General.pth"),
    'Narrabeen': os.path.join(BASE_PATH, "Narrabeen.pth"),
    'Gold Coast': os.path.join(BASE_PATH, "GoldCoast.pth"),
    'CoastSnap': os.path.join(BASE_PATH, "CoastSnap.pth")
}

# MODEL_PATHS = {
#     'General': "/home/yiting/coaste-detect/backend/General.pth",
#     'Narrabeen': "/home/yiting/coaste-detect/backend/Narrabeen.pth",
#     'Gold Coast': "/home/yiting/coaste-detect/backend/GoldCoast.pth",
#     'CoastSnap': "/home/yiting/coaste-detect/backend/CoastSnap.pth"
# }

MODEL_URLS = {
    'General': "https://drive.google.com/uc?id=1gQ7OCHDzCqruQuIzFE2oKAdzsSqsUAkH",
    'Narrabeen': "https://drive.google.com/uc?id=1eJevePtuSCtR7TGT2pidZ4qlRQLlWNMs",
    'Gold Coast': "https://drive.google.com/uc?id=17eTT7zOC9CLfuTBx43pUAQ6JFG7autLl",
    'CoastSnap': "https://drive.google.com/uc?id=1iAT9LjHjYXJvL7iiI3cWYxPI5bWR881V",
    'coast_classifier': "https://drive.google.com/uc?id=1q-Mmf2RFZ7nuNkJy4bNCVgtpxkMK6r2N"
}



def check_and_download_models():
    status = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            logger.info(f"Model {name} is missing. you need to download it.")
            status[name] = "Downloading"
        else:
            status[name] = "Already exists"
            logger.info(f"Model {name} already exists.")

    return status