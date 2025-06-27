import os
import re
import time
import asyncio
import random
import logging
import datetime
import webbrowser
import argparse
import platform
from difflib import get_close_matches
from typing import List

try:
    import easyocr
    import torch
    from PIL import Image
    import numpy as np
    import httpx
    from tqdm import tqdm
    import colorlog
    import db
except ImportError as e:
    logging.error(f"Error importing dependencies: {e} | Attempting auto repair...")
    try:
        import subprocess, requests

        if not os.path.exists("db.py"):
            db_code = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/db.py").text
            with open("db.py", "w") as f:
                f.write(db_code)

        if not os.path.exists("requirements.txt"):
            reqs = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/requirements.txt").text
            with open("requirements.txt", "w") as f:
                f.write(reqs)

        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        import easyocr, torch, httpx, colorlog, db
        from cachetools import TTLCache
        from PIL import Image
        import numpy as np
        from tqdm import tqdm
    except Exception as e:
        logging.error(f"Auto repair failed: {e}")
        exit(1)
except KeyboardInterrupt:
    exit(0)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(reset)s %(message)s"))
logger = colorlog.getLogger("inventory")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.warning("This is experimental software.")

database = db.Database()
item_names: List[str] = []

async def check_ack():
    await database.start()
    if not await database.is_acknowledged():
        if input("Do you understand this? (yes/no): ").lower() in ("y", "yes"):
            await database.acknowledge()
            logger.info("Acknowledged.")
            return True
        logger.error("User aborted.")
        return False
    return True

def match_item(text: str, known: List[str], threshold: float = 0.65):
    match = get_close_matches(text, known, n=1, cutoff=threshold)
    return match[0] if match else None

def format_time(ts):
    d = datetime.datetime.fromtimestamp(ts)
    now = datetime.datetime.now()
    delta = now - d

    if delta.total_seconds() < 60:
        ago = f"{int(delta.total_seconds())} seconds ago"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() // 60)
        ago = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() // 3600)
        ago = f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(delta.total_seconds() // 86400)
        ago = f"{days} day{'s' if days != 1 else ''} ago"

    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    formatted = d.strftime(f"%B {d.day}{suffix} %Y at %I:%M %p")
    return f"{formatted} | {ago}"

async def fetch_items():
    original_names = []
    stripped_names = []
    async with httpx.AsyncClient() as client:
        items = (await client.get('https://api.jailbreakchangelogs.xyz/items/list')).json()
        for i in tqdm(items, desc="Fetching", colour="green"):
            await database.save_item(i['name'], i, int(time.time()))
            original = i['name']
            stripped = original.replace(" ", "").upper()
            original_names.append(original)
            stripped_names.append(stripped)
            time.sleep(random.uniform(0.005, 0.01))
        logger.info(f"Fetched {len(original_names)} items")
        await database.insert(original_names, stripped_names, int(time.time()))
        global item_names
        item_names = stripped_names
        return stripped_names



async def fetch_item(name: str):
    _name = await database.fetch_original_name(name)
    item, last_updated = await database.fetch_item(_name)
    if not item or int(time.time()) - last_updated >= 600:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'https://api.jailbreakchangelogs.xyz/items/get?name={_name}')
            item_list = response.json()
            if not item_list:
                return None
            item = item_list[0]
            await database.save_item(_name, item, int(time.time()))
    return item

async def fetch_cached():
    last = await database.get_last_updated()
    if last is None:
        return await fetch_items()
    logger.info(f"Database last updated: {format_time(last)}")
    if last is None or int(time.time()) - last >= 300:
        logger.info("Updating database...")
        return await fetch_items()

    rows = await database.fetch_names_only()
    if rows:
        global item_names
        item_names = rows
        return item_names
    return await fetch_items()

def is_dark_color(bbox, image: Image.Image, brightness_threshold=100):
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    crop = image.crop((min(xs), min(ys), max(xs), max(ys)))
    arr = np.array(crop)

    if arr.size == 0:
        return False

    avg_rgb = np.mean(arr.reshape(-1, 3), axis=0)
    r, g, b = avg_rgb
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness > brightness_threshold  

async def extract_items(image_path=None):
    if not item_names:
        await fetch_cached()

    image_path = image_path or input("Image path: ")
    if not os.path.exists(image_path):
        logger.error("Image not found.")
        return [], []

    img = Image.open(image_path).convert("RGB")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), model_storage_directory='./models')

    try:
        results = reader.readtext(image_path)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return [], []

    skip = [
        r'COLOR', r'INTRODUCTION', r'INTEGRATION', r'LIST', r'WAITING',
        r'TRADE', r'CHAT', r'DECLINE', r'ACCEPT',
        r'EVERYDAY.*FUEL', r'ZOO\s?PM', r'\d{1,2}:\d{2}\s?[AP]M',
        r'^[A-Z]{1,2}$'
    ]

    detected_items = []
    for bbox, text, _ in results:
        if not is_dark_color(bbox, img):
            continue
        t = re.sub(r'[^A-Z0-9\s]', '', text.strip().upper())
        t = re.sub(r'\b(LVL[IT]|LVLT|LVLI|LEVEL)\b', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        if len(t) < 3 or any(re.search(p, t, re.IGNORECASE) for p in skip):
            continue
        if t.startswith('HYPER'):
            color = t[5:].strip()
            if color:
                t = f'HYPER{color}'
        matched = match_item(t, item_names)
        if matched:
            y_pos = min(point[1] for point in bbox)
            detected_items.append((matched, y_pos))

    if not detected_items:
        return [], []

    detected_items.sort(key=lambda x: x[1])
    y_positions = [y for _, y in detected_items]
    median_y = np.median(y_positions)
    top_row = [item for item, y in detected_items if y < median_y]
    bottom_row = [item for item, y in detected_items if y >= median_y]

    def remove_duplicates(items):
        seen = set()
        return [x for x in items if not (x in seen or seen.add(x))]

    return remove_duplicates(top_row), remove_duplicates(bottom_row)

async def main():
    parser = argparse.ArgumentParser(description='Process inventory image.')
    parser.add_argument('--image', help='Path to the image file to process')
    args = parser.parse_args()

    if await check_ack():
        logger.info("Starting...")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
        else:
            os_name = platform.system()       
            arch = platform.machine()
            if arch == "AMD64": arch = "x86_64"
            os_version = platform.release()  
            url = f"https://developer.nvidia.com/cuda-downloads?target_os={os_name}&target_arch={arch}&target_version={os_version}&target_type=exe_local"
            logger.warning("CUDA not found.")
            logger.warning("CUDA uses your graphics card to run faster. You can still use this program without a GPU, but it will be slower.")
            confirmation = input("Would you like to be sent to the download page? (yes/no): ").lower()
            if confirmation in ("y", "yes"):
                webbrowser.open(url)
                logger.info("Opening download page...")
                return
            else:
                logger.info("Skipping download page...")

        try:
            top, bottom = await extract_items(args.image)
            if top or bottom:
                logger.info("Top row items:")
                for item in top:
                    _item = await fetch_item(item)
                    if _item is None:
                        logger.warning(f"Item not found: {item}")
                    else:
                        logger.info(f"{_item['name']} - {_item['type']} | Cash Value: {_item['cash_value']} Duped Value: {_item['duped_value']}")
                logger.info("Bottom row items:")
                for item in bottom:
                    _item = await fetch_item(item)
                    if _item is None:
                        logger.warning(f"Item not found: {item}")
                    else:
                        logger.info(f"{_item['name']} - {_item['type']} | Cash Value: {_item['cash_value']} Duped Value: {_item['duped_value']}")
                logger.info("Thats all! More features will be added in the future.")
        except KeyboardInterrupt:
            logger.info("Exiting...")
            exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
