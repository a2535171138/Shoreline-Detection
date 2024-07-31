import os
import gdown
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATHS = {
    'General': "/home/yiting/coaste-detect/backend/General.pth",
    'Narrabeen': "/home/yiting/coaste-detect/backend/Narrabeen.pth",
    'Gold Coast': "/home/yiting/coaste-detect/backend/GoldCoast.pth",
    'CoastSnap': "/home/yiting/coaste-detect/backend/CoastSnap.pth",
    'coast_classifier': "/home/yiting/coaste-detect/backend/coast_classifier.pth"
}

MODEL_URLS = {
    'General': "https://drive.google.com/uc?id=1gQ7OCHDzCqruQuIzFE2oKAdzsSqsUAkH",
    'Narrabeen': "https://drive.google.com/uc?id=1eJevePtuSCtR7TGT2pidZ4qlRQLlWNMs",
    'Gold Coast': "https://drive.google.com/uc?id=17eTT7zOC9CLfuTBx43pUAQ6JFG7autLl",
    'CoastSnap': "https://drive.google.com/uc?id=1iAT9LjHjYXJvL7iiI3cWYxPI5bWR881V",
    'coast_classifier': "https://drive.google.com/uc?id=1q-Mmf2RFZ7nuNkJy4bNCVgtpxkMK6r2N"
}


def download_file(url, output, status, name):
    try:
        gdown.download(url, output, quiet=False)
        logger.info(f"Successfully downloaded {output}")
        status[name] = "Downloaded"
    except Exception as e:
        logger.error(f"Failed to download {output}: {str(e)}")
        status[name] = "Download failed"


def check_and_download_models():
    status = {}
    threads = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            logger.info(f"Model {name} is missing. Starting download...")
            status[name] = "Downloading"
            thread = threading.Thread(target=download_file, args=(MODEL_URLS[name], path, status, name))
            thread.start()
            threads.append(thread)
        else:
            status[name] = "Already exists"
            logger.info(f"Model {name} already exists.")

    for thread in threads:
        thread.join()

    return status