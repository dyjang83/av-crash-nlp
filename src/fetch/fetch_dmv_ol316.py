"""Fetch California DMV OL 316 autonomous-vehicle collision report PDFs.

The DMV collision-reports listing is an IBM WebSphere portal page that renders
links via JavaScript, sometimes inside iframes. Even published pipelines (e.g.
manimalakumar/Autonomous-Vehicle-CA-DMV) instruct users to download the PDFs by
hand. This module automates it where possible and otherwise gives you a reliable
manual path plus diagnostics so the harvester can be tuned to the live markup.

Confirmed individual-report URL forms (June 2026), all under /portal/file/:
    .../portal/file/cruise_083021/
    .../portal/file/zoox_12082025-pdf/
    .../portal/file/collision-report-zoox-january-21-2020-2-pdf/
    .../portal/file/waymo-collision-report-august-9-2019-2-pdf/
Note: slugs may use either `<mfr>_<date>` or `collision-report-<mfr>-<date>` and
may or may not end in `-pdf`.

Acquisition modes:
  --render [--debug]   Playwright renders the page (and all iframes), scrolls to
                       trigger lazy loads, harvests report links, and on --debug
                       dumps the rendered HTML to data/raw/dmv_listing_rendered.html
                       plus a breakdown of every link it saw (so you can show the
                       structure or feed the dump to --from-html).
  --from-html FILE     Parse links from a saved, fully-rendered page (Save Page
                       As "Webpage, Complete" in your browser). Always works.
  --links-file FILE    Plain text file of report URLs, one per line.
  (default)            Static fetch + broadened matcher; reports 0 + guidance if
                       the page is JS-gated.

Usage:
    python -m src.fetch.fetch_dmv_ol316 --render --debug
    python -m src.fetch.fetch_dmv_ol316 --from-html dmv_listing_rendered.html
    python -m src.fetch.fetch_dmv_ol316 --links-file urls.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from urllib.parse import urljoin

import requests

LISTING = ("https://www.dmv.ca.gov/portal/vehicle-industry-services/"
           "autonomous-vehicles/autonomous-vehicle-collision-reports/")
BASE = "https://www.dmv.ca.gov"
WP_API = f"{BASE}/portal/wp-json"
WP_PAGE_SLUG = "autonomous-vehicle-collision-reports"
OUT_DIR = os.path.join("data", "raw", "ol316")
DUMP_HTML = os.path.join("data", "raw", "dmv_listing_rendered.html")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}
SLEEP_S = 1.0

ANY_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)

EXCLUDE_SUBSTR = (
    "report-of-traffic-accident-involving-an-autonomous-vehicle-ol-316",  # blank form
    "report-of-traffic-collision-involving-an-autonomous-vehicle-ol-316",
    "disengagement", "ol-311", "ol311", "mileage", "permit-holder",
    "regulation", "instructions", "-form-", "template", "deployment",
    "ol-316-pdf", "ol316-pdf",
)
MANUFACTURERS = ("waymo", "cruise", "zoox", "nuro", "apple", "pony", "aurora",
                 "mercedes", "tesla", "gatik", "motional", "wayve", "didi",
                 "argo", "lyft", "uber", "deeproute", "autox", "valeo", "toyota",
                 "nissan", "bmw", "ghost", "imagry", "qcraft", "weride", "kodiak",
                 "nvidia", "ridecell", "udelv", "cyngn", "easymile", "navya")
DATEISH = re.compile(r"\d{4,8}|\d{1,2}[-_]\d{1,2}[-_]\d{2,4}|"
                     r"(january|february|march|april|may|june|july|august|"
                     r"september|october|november|december|"
                     r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", re.I)


def _slug_of(url: str) -> str:
    return url.rstrip("/").split("/")[-1].lower()


def _looks_like_report(url: str) -> bool:
    s = _slug_of(url)
    if any(x in url.lower() for x in EXCLUDE_SUBSTR):
        return False
    has_mfr = any(m in s for m in MANUFACTURERS)
    if has_mfr and ("collision" in s or "report" in s or DATEISH.search(s)
                    or "_" in s):
        return True
    if "collision-report" in s and DATEISH.search(s):
        return True
    return False


def harvest_from_html(html: str, base_url: str = BASE,
                      debug: bool = False) -> list[str]:
    hrefs = [urljoin(base_url, h) for h in ANY_HREF.findall(html)]
    reports = sorted({u for u in hrefs if _looks_like_report(u)})
    if debug:
        portal_file = [u for u in hrefs if "/portal/file/" in u]
        prefixes = Counter()
        for u in hrefs:
            m = re.match(r"https?://[^/]+(/[^/]*/?[^/]*)", u)
            prefixes[m.group(1) if m else u[:40]] += 1
        print(f"[debug] total hrefs: {len(hrefs)}")
        print(f"[debug] /portal/file/ links: {len(portal_file)}")
        print(f"[debug] report-like links: {len(reports)}")
        print("[debug] top href path-prefixes:")
        for pre, c in prefixes.most_common(15):
            print(f"          {c:4d}  {pre}")
        print("[debug] sample hrefs:")
        for u in hrefs[:30]:
            print(f"          {u}")
    print(f"[ol316] harvested {len(hrefs)} links, "
          f"{len(reports)} look like collision reports")
    return reports


def harvest_wp_json(debug: bool = False) -> list[str]:
    """Query the WordPress REST API instead of scraping rendered HTML.

    The DMV portal is WordPress (wp-json/elasticpress). The collision-report list
    is injected client-side, so it never appears in scraped HTML -- but the
    underlying data is reachable via the REST API. Strategy, in order:
      1. Fetch the page by slug and harvest links from its authored content.
      2. Page through the media library for collision-report PDFs.
      3. (debug) Print available REST routes so the endpoint can be pinned down.
    """
    urls: list[str] = []

    # 1) Page content by slug -> content.rendered usually holds the authored links.
    try:
        r = requests.get(f"{WP_API}/wp/v2/pages",
                         params={"slug": WP_PAGE_SLUG, "_fields": "id,link,content"},
                         headers=HEADERS, timeout=60)
        if r.status_code == 200:
            for pg in r.json():
                content = (pg.get("content") or {}).get("rendered", "")
                if debug:
                    print(f"[wp-json] page content.rendered: {len(content):,} chars")
                urls += harvest_from_html(content, debug=debug)
        else:
            print(f"[wp-json] pages query HTTP {r.status_code}")
    except requests.RequestException as e:
        print(f"[wp-json] pages query failed: {e}")

    # 2) Media library PDFs (paginated), filtered to report-like source URLs.
    if not urls:
        try:
            page = 1
            while page <= 80:
                r = requests.get(f"{WP_API}/wp/v2/media",
                                 params={"per_page": 100, "page": page,
                                         "mime_type": "application/pdf",
                                         "_fields": "source_url,slug"},
                                 headers=HEADERS, timeout=60)
                if r.status_code != 200:
                    if debug:
                        print(f"[wp-json] media page {page} HTTP {r.status_code}")
                    break
                items = r.json()
                if not items:
                    break
                urls += [it["source_url"] for it in items
                         if _looks_like_report(it.get("source_url", ""))]
                total = int(r.headers.get("X-WP-TotalPages", page))
                if page >= total:
                    break
                page += 1
            if debug:
                print(f"[wp-json] media scan kept {len(urls)} report-like PDF(s)")
        except requests.RequestException as e:
            print(f"[wp-json] media query failed: {e}")

    # 3) Route discovery to guide further tuning if nothing matched.
    if not urls and debug:
        try:
            routes = requests.get(WP_API, headers=HEADERS, timeout=60).json()
            print("[wp-json] available REST routes:")
            for rt in sorted(routes.get("routes", {}).keys()):
                print("   ", rt)
        except requests.RequestException as e:
            print(f"[wp-json] route discovery failed: {e}")

    out = sorted(set(urls))
    print(f"[ol316] wp-json harvested {len(out)} collision-report link(s)")
    return out


def harvest_static(debug: bool = False) -> list[str]:
    try:
        html = requests.get(LISTING, headers=HEADERS, timeout=60).text
    except requests.RequestException as e:
        print(f"[ol316] static fetch failed: {e}", file=sys.stderr)
        return []
    return harvest_from_html(html, debug=debug)


def harvest_render(debug: bool = False) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("[ol316] --render needs Playwright: pip install playwright && "
                 "playwright install chromium")
    htmls = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(LISTING, wait_until="networkidle", timeout=90_000)
        # Trigger lazy content: scroll to bottom a few times, then wait.
        for _ in range(6):
            page.mouse.wheel(0, 20_000)
            page.wait_for_timeout(800)
        try:
            page.wait_for_selector("a[href*='/portal/file/']", timeout=8_000)
        except Exception:  # noqa: BLE001 - selector may legitimately be absent
            pass
        # Expand accordions / click year or manufacturer toggles if present.
        for sel in ["button[aria-expanded='false']", ".accordion__header",
                    "summary", "[role='button']"]:
            for el in page.query_selector_all(sel):
                try:
                    el.click(timeout=1000)
                    page.wait_for_timeout(200)
                except Exception:  # noqa: BLE001
                    pass
        # Paginate.
        for _ in range(60):
            nxt = page.query_selector("a[rel='next'], .pager__item--next a, "
                                      "a[aria-label='Next']")
            if not nxt:
                break
            try:
                nxt.click()
                page.wait_for_load_state("networkidle", timeout=30_000)
            except Exception:  # noqa: BLE001
                break
        # Harvest from main frame AND every child frame (portal content is often
        # rendered inside iframes, which is the usual reason a render finds zero).
        for fr in page.frames:
            try:
                htmls.append(fr.content())
            except Exception:  # noqa: BLE001
                pass
        browser.close()

    merged = "\n".join(htmls)
    if debug:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(DUMP_HTML, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"[debug] wrote rendered HTML ({len(merged):,} chars, "
              f"{len(htmls)} frame(s)) to {DUMP_HTML}")
    return harvest_from_html(merged, debug=debug)


def download(urls: list[str]) -> list[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for i, url in enumerate(urls):
        fn = os.path.join(OUT_DIR, f"{_slug_of(url)}.pdf")
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            paths.append(fn)
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            if r.content[:4] != b"%PDF" and "pdf" not in \
                    r.headers.get("content-type", "").lower():
                print(f"[ol316] skip non-PDF: {url}")
                continue
            with open(fn, "wb") as f:
                f.write(r.content)
            paths.append(fn)
            print(f"[ol316] ({i+1}/{len(urls)}) {fn} ({len(r.content):,} bytes)")
        except requests.HTTPError as e:
            print(f"[ol316] skip {url}: {e}")
        time.sleep(SLEEP_S)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Fetch CA DMV OL 316 report PDFs.")
    ap.add_argument("--wp-json", action="store_true",
                    help="Query the WordPress REST API (recommended; no scraping).")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--from-html", metavar="FILE")
    ap.add_argument("--links-file", metavar="FILE")
    ap.add_argument("--debug", action="store_true",
                    help="Dump rendered HTML + link diagnostics.")
    args = ap.parse_args()

    if args.links_file:
        with open(args.links_file) as f:
            urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        print(f"[ol316] {len(urls)} URLs from {args.links_file}")
    elif args.from_html:
        with open(args.from_html, encoding="utf-8", errors="ignore") as f:
            urls = harvest_from_html(f.read(), debug=args.debug)
    elif args.render:
        urls = harvest_render(debug=args.debug)
    elif args.wp_json:
        urls = harvest_wp_json(debug=args.debug)
    else:
        # Default: try the REST API first (this portal is WordPress), then a
        # plain static fetch as a secondary attempt.
        urls = harvest_wp_json(debug=args.debug) or harvest_static(debug=args.debug)

    if not urls:
        print("\n[ol316] No report links found.\n"
              "This portal is WordPress; the report list is injected client-side,\n"
              "so HTML scraping sees only the page shell. Try, in order:\n"
              "  1. python -m src.fetch.fetch_dmv_ol316 --wp-json --debug\n"
              "     (queries the REST API and, if needed, prints available routes)\n"
              "  2. Open the page in your browser, let the list render, Save Page\n"
              "     As 'Webpage, Complete', then --from-html that file.\n"
              "If --wp-json --debug prints REST routes, paste them and the media/\n"
              "page endpoint can be pinned exactly.\n", file=sys.stderr)
        sys.exit(1)

    paths = download(urls)
    print(f"\n[ol316] done. {len(paths)} PDF(s) in {OUT_DIR}/")


if __name__ == "__main__":
    main()