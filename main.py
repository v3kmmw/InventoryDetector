import os
import re
import time
import asyncio
import random
import logging
import datetime
from difflib import get_close_matches
from typing import List

try:
    import easyocr
    import torch
    import httpx
    from tqdm import tqdm
    import colorlog
    import db
except ImportError as e:
    print(f"Failed to import: {e}, attempting auto-repair...")
    import subprocess, requests

    if not os.path.exists("db.py"):
        print("Downloading db.py...")
        db_code = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/db.py").text
        with open("db.py", "w") as f:
            f.write(db_code)

    if not os.path.exists("requirements.txt"):
        print("Downloading requirements.txt...")
        reqs = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/requirements.txt").text
        with open("requirements.txt", "w") as f:
            f.write(reqs)

    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
    import easyocr, torch, httpx, colorlog, db
    from tqdm import tqdm

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

def match_item(text: str, known: List[str], threshold: float = 0.7):
    match = get_close_matches(text, known, n=1, cutoff=threshold)
    return match[0] if match else None

def format_time(ts):
    d = datetime.datetime.fromtimestamp(ts)
    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return d.strftime(f"%B {d.day}{suffix} %Y at %I:%M %p")

async def fetch_items():
    async with httpx.AsyncClient() as client:
        items = (await client.get('https://api.jailbreakchangelogs.xyz/items/list')).json()
        for i in tqdm(items, desc="Fetching", colour="green"):
            item_names.append(i['name'].replace(" ", "").upper())
            time.sleep(random.uniform(0.005, 0.01))
        logger.info(f"Fetched {len(item_names)} items")
        await database.insert(item_names, int(time.time()))
        return item_names

async def fetch_cached():
    last = await database.get_last_updated()
    if last is None or int(time.time()) - last >= 60:
        logger.info("Updating database...")
        return await fetch_items()

    logger.info(f"DB last updated: {format_time(last)}")
    rows = await database.fetch_names_only()
    if rows:
        global item_names
        item_names = rows
        return item_names
    return await fetch_items()

async def extract_items(image_path=None):
    if not item_names:
        await fetch_cached()

    image_path = image_path or input("Image path: ")
    if not os.path.exists(image_path):
        logger.error("Image not found.")
        return

    reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./models')
    try:
        results = reader.readtext(image_path)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return

    skip = [r'COLOR', r'INTRODUCTION', r'TRADE', r'CHAT', r'DECLINE', r'ACCEPT', r'EVERYDAY.*FUEL', r'ZOO PM', r'\d{1,2}:\d{2} [AP]M']
    items = []
    for _, text, _ in results:
        t = re.sub(r'[^A-Z]', '', text.strip().upper())
        t = re.sub(r'\bLVL[IT]\b', '', t)
        t = t.replace('VEH', 'VEHICLE')
        if len(t) >= 3 and not any(re.search(p, t) for p in skip):
            items.append(match_item(t, item_names))
    return [i for i in items if i]

async def main():
    if await check_ack():
        logger.info("Starting...")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")

        results = await extract_items()
        if results:
            print("\n".join(results))

if __name__ == "__main__":
    asyncio.run(main())
