
try:
    import easyocr 
    import re
    from typing import List, Tuple
    from difflib import get_close_matches
    import torch
    import asyncio
    import random
    import os
    import httpx
    from tqdm import tqdm
    import time
    import datetime
    import db
    import logging
    import colorlog
except ImportError as e:
    print(f"Failed to import: {e}, trying to redownload requirements...")
    import os
    if not os.path.exists("db.py"):
        print("Downloading database...")
        import requests
        req = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/db.py")
        with open("db.py", "w") as f:
            f.write(req.text)
    if not os.path.exists("requirements.txt"):
        print("Downloading requirements.txt...")
        import requests
        req = requests.get("https://raw.githubusercontent.com/v3kmmw/InventoryDetector/refs/heads/main/requirements.txt")
        with open("requirements.txt", "w") as f:
            f.write(req.text)
    import subprocess

    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
    import easyocr 
    import re
    import os
    from typing import List, Tuple
    from difflib import get_close_matches
    import torch
    import asyncio
    import random
    import httpx
    import datetime
    import db
    from tqdm import tqdm
    import time
    import logging
    import colorlog


handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(reset)s %(message)s"
))

logger = colorlog.getLogger("mylogger")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.warning("This is experimental, it may not work as expected!")
acknowledged = False

database = db.Database()

def ask_for_confirmation(message: str) -> bool:
    while True:
        user_input = input(f"{message} (yes/no): ").lower()
        if user_input in {"yes", "y"}:
            return True
        elif user_input in {"no", "n"}:
            return False
        else:
            logger.warning("Invalid input. Please enter 'yes' or 'no'.")

async def check_and_acknowledge():
    await database.start()
    if not await database.is_acknowledged():
        if ask_for_confirmation("Do you understand this?"):
            await database.acknowledge()
            logger.info("Acknowledged. Proceeding...")
            return True
        else:
            logger.error("Aborting due to lack of confirmation.")
            return False
    else:
        return True

item_names = []

async def fetch_items():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://api.jailbreakchangelogs.xyz/items/list')
        items = response.json()
        for item in tqdm(
            items,
            desc="Fetching",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green",
            smoothing=0.1
        ):
            item_names.append(item['name'].replace(" ", "").upper())
            time.sleep(random.uniform(0.005, 0.01))
        logger.info(f"Successfully fetched {len(item_names)} item names")
        await database.insert(item_names, int(time.time()))
        return item_names

def match_closest_item(text: str, known_items: list[str], threshold: float = 0.7) -> str:
    matches = get_close_matches(text, known_items, n=1, cutoff=threshold)
    return matches[0] if matches else None
        
def format_datetime(ts):
    dt = datetime.datetime.fromtimestamp(ts)
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1:"st", 2:"nd", 3:"rd"}.get(day % 10, "th")
    return dt.strftime(f"%B {day}{suffix} %Y at %I:%M %p")
    
async def fetch_raw_item_names():
    last_updated = await database.get_last_updated()

    if last_updated is None:
        await fetch_items()
        return await fetch_raw_item_names()

    logger.info("Database last updated: %s", format_datetime(last_updated))
    if int(time.time()) - last_updated < 60: 
        item_rows = await database.fetch_names_only()
        if item_rows:
            global item_names
            item_names = item_rows
            return item_names
        else:
            logger.info("Database is empty, filling it up...")
            return await fetch_items()
            
    else:
        logger.info("Database is out of date, updating...")
        return await fetch_items()
           


async def extract_and_pair_items(image_path: str = None) -> List[Tuple[str, str]]:
    tries = 0
    while len(item_names) == 0 and tries < 3: 
        tries += 1
        if tries > 2:
            logger.error("Failed to fetch items, aborting...")
            return 
        items = await fetch_raw_item_names()
        time.sleep(2)
        if not items:
            logger.error("Failed to fetch items, retrying...")
            return
    
    # ask for the image
    if not image_path:
        image_path = input("Please provide the path to the image: ")
    if not os.path.exists(image_path):
        logger.error("Image not found.")
        return
        
    """Extract and pair trade items from an image with improved pairing logic."""
    reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./models')
    try:
        results = reader.readtext(image_path)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return
    # Enhanced skip patterns (using regex for better matching)
    skip_patterns = [
        r'COLOR', r'INTRODUCTION', r'INTEGRATION', r'LIST', 
        r'TRADE', r'CHAT', r'DECLINE', r'ACCEPT',
        r'EVERYDAY.*FUEL', r'ZOO PM', r'\d{1,2}:\d{2} [AP]M'
    ]
    
    items = []
    for (bbox, text, prob) in results:
        text = text.strip().upper()
        
        # Skip if matches any skip pattern
        if any(re.search(pattern, text) for pattern in skip_patterns):
            continue
            
        # Enhanced cleaning
        text = re.sub(r'[^A-Z]', '', text)  # Remove all non-alphabet characters
        text = re.sub(r'\bLVL[IT]\b', '', text)  # Remove level indicators
        text = re.sub(r'\bVEH\b', 'VEHICLE', text)  # Normalize vehicle
        
        if len(text) >= 3:
            matched = match_closest_item(text, item_names)
            items.append(matched)

    return items

def format_pairs(pairs: List[Tuple[str, str]]) -> str:
    """Format the pairs for display."""
    return "\n".join(f"{item}" for item in pairs if item)

async def main():
    if await check_and_acknowledge():
        logger.info("Starting...")
        if torch.cuda.is_available(): 
            torch.cuda.set_device(0) 
            torch.cuda.empty_cache()
            logger.info(f"CUDA is available, using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available, using CPU | This may be slower!")
            
        pairs = await extract_and_pair_items()
        if pairs:
            print("Detected item pairs:")
            print(format_pairs(pairs))
    else:
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
