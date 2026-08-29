"""Fetch NHTSA Standing General Order (SGO 2021-01) incident-report CSVs.

NHTSA publishes the SGO crash data as static CSVs at known permanent URLs.
There are two sets:
  - Current (third amendment, June 16 2025 onward): ADS, ADAS, OTHER
  - Archive (prior versions, 2021-2025): ADS, ADAS, OTHER

Both are downloaded by default so the full longitudinal corpus is available.
The page does NOT surface CSV links via JavaScript-rendered content; the URLs
are hardcoded directly (confirmed June 2026 from the SGO landing page).

Usage:
    python -m src.fetch.fetch_sgo
    python -m src.fetch.fetch_sgo --archive-only
    python -m src.fetch.fetch_sgo --current-only
"""
from __future__ import annotations

import argparse
import os
import sys
import requests

HEADERS = {"User-Agent": "av-crash-nlp research fetcher (contact: dyjang83@github)"}
OUT_DIR = os.path.join("data", "raw", "sgo")

# Third amendment: June 16 2025 → present
CURRENT = {
    "SGO-2021-01_Incident_Reports_ADS.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv",
    "SGO-2021-01_Incident_Reports_ADAS.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv",
    "SGO-2021-01_Incident_Reports_OTHER.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_OTHER.csv",
}

# Archive: prior versions (2021-2025)
ARCHIVE = {
    "Archive_SGO-2021-01_Incident_Reports_ADS.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADS.csv",
    "Archive_SGO-2021-01_Incident_Reports_ADAS.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADAS.csv",
    "Archive_SGO-2021-01_Incident_Reports_OTHER.csv":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_OTHER.csv",
}

# Data element definitions PDF (optional, for reference)
DEFINITIONS_PDF = {
    "SGO-2021-01_Data_Element_Definitions.pdf":
        "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Data_Element_Definitions.pdf",
}


def download(files: dict[str, str], skip_existing: bool = True) -> list[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for filename, url in files.items():
        dest = os.path.join(OUT_DIR, filename)
        if skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[sgo] skip (exists): {dest}")
            paths.append(dest)
            continue
        print(f"[sgo] downloading {filename} ...")
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            size = os.path.getsize(dest)
            print(f"[sgo] -> {dest} ({size:,} bytes)")
            paths.append(dest)
        except requests.HTTPError as e:
            print(f"[sgo] ERROR {url}: {e}", file=sys.stderr)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Fetch NHTSA SGO incident report CSVs.")
    ap.add_argument("--current-only", action="store_true",
                    help="Download only the current (2025-present) third-amendment files.")
    ap.add_argument("--archive-only", action="store_true",
                    help="Download only the archive (2021-2025) files.")
    ap.add_argument("--no-pdf", action="store_true",
                    help="Skip the data-element-definitions PDF.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if local file already exists.")
    args = ap.parse_args()

    to_download = {}
    if not args.archive_only:
        to_download.update(CURRENT)
    if not args.current_only:
        to_download.update(ARCHIVE)
    if not args.no_pdf:
        to_download.update(DEFINITIONS_PDF)

    paths = download(to_download, skip_existing=not args.force)
    csv_paths = [p for p in paths if p.endswith(".csv")]
    print(f"\n[sgo] done. {len(csv_paths)} CSV file(s) in {OUT_DIR}/")
    if not csv_paths:
        print("[sgo] No files downloaded. Check your network connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()