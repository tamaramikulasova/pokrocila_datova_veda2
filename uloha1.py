from __future__ import annotations


import os
import re
import shutil
import json
import warnings
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import math
import random

import requests
from duckduckgo_search import DDGS
from tqdm import tqdm


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters for filenames."""
    name = re.sub(r"[^\w\-. ]", "_", name)
    name = name.strip().strip(".")
    return name or "image"


def search_images(keyword: str, max_results: int = 10) -> List[str]:
    """
    Returns a list of direct image URLs from DuckDuckGo.
    """
    # DDGS().images returns a generator; convert to list then slice.
    with DDGS() as ddgs:
        results = list(
            ddgs.images(
                keywords=keyword,
                region="wt-wt",         # worldwide
                safesearch="off",       # "off", "moderate", "strict"
                size=None,              # "Large", "Medium", "Small"
                color=None,             # e.g., "color", "Monochrome"
                type_image=None,        # "photo", "clipart", "gif", ...
                layout=None,
                license_image=None,
            )
        )

    urls = []
    # Some items use "image", others "image_url" depending on ddgs version
    for r in results:
        u = r.get("image") or r.get("image_url") or r.get("thumbnail")
        if u != []:
            urls.append(u)

    return urls[:max_results]


def download_image(
    url: str,
    folder: str | Path,
    custom_name: Optional[str] = None,
    verbose: bool = True,
    timeout: int = 15,
) -> bool:
    """
    Download one image (if Content-Type is image/*) into `folder`.
    Auto-add extension when missing and avoid overwriting by adding a counter.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Filename: custom_name (sanitized) or derived from URL path
    if custom_name:
        filename = sanitize_filename(custom_name)
    else:
        path = Path(urlparse(url).path)
        filename = sanitize_filename(path.name or "image.jpg")

    # Ensure it has an extension (default .jpg)
    if not Path(filename).suffix:
        filename += ".jpg"

    filepath = folder / filename

    # Avoid overwriting existing file
    base = filepath.with_suffix("")
    ext = filepath.suffix
    counter = 1
    while filepath.exists():
        filepath = Path(f"{base}_{counter}{ext}")
        counter += 1

    try:
        # Stream to avoid loading big files fully in memory
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageFetcher/1.0)"}
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image"):
                if verbose:
                    warnings.warn(
                        f"Skipped (not an image): {url} (Content-Type: {content_type})"
                    )
                return False

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if verbose:
            print(f"✓ Saved: {filepath}")
        return True

    except requests.exceptions.Timeout:
        if verbose:
            warnings.warn(f"Timeout: {url}")
    except requests.exceptions.HTTPError as e:
        if verbose:
            warnings.warn(f"HTTP error: {e} ({url})")
    except requests.exceptions.RequestException as e:
        if verbose:
            warnings.warn(f"Request error: {e} ({url})")
    except OSError as e:
        if verbose:
            warnings.warn(f"I/O error: {e} ({filepath})")

    return False


if __name__ == "__main__":
    KEYWORDS = ["guitar photo", "trumpet photo",  "piano photo", "drums photo"]
    MAX_RESULTS = 250           # change as you like
    OUT_DIR = Path("./dataset/guitar")

    with open("idk.json", "r") as file:
        dict_urls = json.load(file)

    try:
        for kw in KEYWORDS:
            if len(dict_urls[kw.split(" ")[0]]) >= 200:
                continue
            a = search_images(kw, MAX_RESULTS)
            dict_urls[kw.split(" ")[0]] += a
            print(f" ========== for {kw} ============")
            print(f"for one ddg: {len(a)}")
            while len(dict_urls[kw.split(" ")[0]]) < 200:
                a = search_images(kw, MAX_RESULTS)
                print(f"for one ddg: {len(a)}")
                dict_urls[kw.split(" ")[0]] += a
        with open(f"idk.json", "w") as file:
            json.dump(dict_urls, file, indent=4)
    except:
        print("=== SAVING ===")
        with open(f"idk.json", "w") as file:
            json.dump(dict_urls, file, indent=4)


    with open("idk.json", "r") as file:
        config = json.load(file)

    test_data = {}
    train_data = {}
    for i in config:
        data = config[i]
        data_l = math.ceil(len(data)*0.25)
        t_tmp = set()
        test_data[i] = data[:data_l]
        train_data[i] = data[data_l:]
    
    with open(f"test_data.json", "w") as file:
        json.dump(test_data, file, indent=4)
    with open(f"train_data.json", "w") as file:
        json.dump(train_data, file, indent=4)
    
    with open("test_data.json", "r") as file:
        test_data = json.load(file)
    with open("train_data.json", "r") as file:
        train_data = json.load(file)

    print("\nDownloading...")
    successes = 0
    for img_key in test_data:
        image_urls = test_data[img_key]
        OUT_DIR = Path(f"./dataset/test_data/{img_key}")
        for i, url in enumerate(tqdm(image_urls, unit="img")):
            ok = download_image(url, OUT_DIR, custom_name=f"image_{i}.jpg", verbose=False)
            successes += int(ok)

        print("\nSaved to:", OUT_DIR)
        print("Files downloaded:", successes)

        # Zip the entire dataset folder (Colab-friendly)
        root = Path("./dataset")
        zip_base = root if root.exists() else OUT_DIR.parent
        zip_name = "dataset"  # produces dataset.zip next to the folder
        print("Zipping folder...", zip_base)

        shutil.make_archive(zip_name, "zip", root_dir=zip_base, base_dir=".")
        print(f"Created archive: {Path(zip_name + '.zip')}")
    for img_key in train_data:
        image_urls = train_data[img_key]
        OUT_DIR = Path(f"./dataset/train_data/{img_key}")
        for i, url in enumerate(tqdm(image_urls, unit="img")):
            ok = download_image(url, OUT_DIR, custom_name=f"image_{i}.jpg", verbose=False)
            successes += int(ok)

        print("\nSaved to:", OUT_DIR)
        print("Files downloaded:", successes)

        # Zip the entire dataset folder (Colab-friendly)
        root = Path("./dataset")
        zip_base = root if root.exists() else OUT_DIR.parent
        zip_name = "dataset"  # produces dataset.zip next to the folder
        print("Zipping folder...", zip_base.resolve())

        shutil.make_archive(zip_name, "zip", root_dir=zip_base, base_dir=".")
        print(f"Created archive: {Path(zip_name + '.zip').resolve()}")

