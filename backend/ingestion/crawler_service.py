"""
Crawler Service Component.

Refactored for High Performance & Reliability:

FIXES APPLIED:
1. URL-agnostic Smart Selection using entropy + similarity.
2. Parallel DB Writer Queue (OPTION A).
3. Thread-safe visited set via asyncio.Lock.
4. Zero-lag persistence (workers never touch SQLite).
5. Safe worker shutdown (stop event).
6. Resource Blocking (Images/Fonts) for speed.
7. Canonical URL handling (Redirects).
8. Content Deduplication (Header/Footer removal).
"""

import asyncio
import os
import json
import urllib.robotparser
import requests
import uuid
import math
from urllib.parse import urlparse, urljoin
from datetime import datetime
from playwright.async_api import async_playwright, Page, BrowserContext
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from backend.database import init_db, insert_page_async, get_all_pages, enable_wal

class CrawlerService:

    def __init__(self):
        init_db()
        enable_wal()
        self.allowed_links = []
        self.blocked_links = []

    # ---------------- SMART URL SCORING (NO KEYWORDS) ---------------- #

    def _url_entropy(self, url: str) -> float:
        probs = [url.count(c)/len(url) for c in set(url)]
        return -sum(p * math.log2(p) for p in probs)

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _get_best_links(self, links: list, visited: set, limit: int = 5) -> list:
        """
        Truly generic smart selection:
        • URL entropy
        • path length
        • depth
        • duplicate similarity
        """

        scored = []

        for link in links:
            u = urlparse(link)
            path = u.path

            depth = len(list(filter(None, path.split("/"))))
            entropy = self._url_entropy(link)
            plen = len(path)

            dup_penalty = 0
            for v in visited:
                if self._similarity(link, v) > 0.85:
                    dup_penalty -= 5

            score = entropy + (2 <= depth <= 4) * 2 + (20 < plen < 120) * 2 + dup_penalty
            scored.append((score, link))

        scored.sort(reverse=True)
        return [x[1] for x in scored[:limit]]

    # ---------------- ROBOTS ---------------- #

    def _get_robots_parser(self, url):
        parsed = urlparse(url)
        robots = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(robots, timeout=5)
            rp.parse(r.text.splitlines())
        except:
            rp.allow_all = True
        return rp

    def is_allowed(self, url, rp):
        try:
            allowed = rp.can_fetch("*", url)
            return allowed
        except:
            return True

    # ---------------- HELPERS ---------------- #

    async def _close_popups(self, page: Page):
        """Attempts to close common cookie banners and modals."""
        selectors = [
            "button[id*='cookie']", "button[class*='cookie']",
            "button[id*='accept']", "button[class*='accept']",
            "button[aria-label*='close']", ".modal-close", "div[aria-label*='cookie'] button",
            "text=Accept All", "text=Agree", "text=No Thanks", "text=Accept"
        ]
        # Quick race to see if any exist, don't wait long
        try:
            for sel in selectors:
                if await page.isVisible(sel, timeout=200):
                    await page.click(sel, timeout=200)
                    break
        except:
            pass

    async def _handle_captcha(self, page: Page):
        """Standardized captcha handling structure."""
        # 1. Cloudflare Turnstile "Verify you are human" checkbox
        try:
            if await page.isVisible("iframe[src*='turnstile']", timeout=1000):
                frames = page.frames
                for f in frames:
                    if "turnstile" in f.url:
                        await f.click("body", timeout=500)
                        await asyncio.sleep(1) # Wait for processing
        except:
            pass

    def _clean_content(self, html: str) -> str:
        """Removes headers, footers, navs to extract main content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove distractions
        for tag in soup(["header", "footer", "nav", "aside", "script", "style", "noscript", "iframe"]):
            tag.decompose()
            
        return soup.get_text(separator="\n", strip=True)

    # ---------------- ASYNC DB WRITER ---------------- #

    async def _db_writer(self, queue):
        """
        Dedicated persistence coroutine.
        Workers push records here.
        This completely removes SQLite from crawler critical path.
        """
        while True:
            item = await queue.get()
            if item is None:
                break
            insert_page_async(*item)
            queue.task_done()

    # ---------------- WORKER ---------------- #

    async def _worker(self, wid, queue, context, session_id, rp,
                      visited, visited_lock, db_queue, simulate, stop_event):

        # Block resources for speed if not in simulation mode
        if not simulate:
            try:
                await context.route("**/*.{png,jpg,jpeg,gif,webp,svg,css,woff,woff2,ttf,mp4,webm,ad,ads}", lambda route: route.abort())
            except:
                pass

        page = await context.new_page()

        while not stop_event.is_set():
            # Get item with timeout to check stop_event frequently
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            
            if item is None:
                queue.task_done()
                break

            url, depth, max_depth = item

            # Check if stopped mid-work
            if stop_event.is_set():
                queue.task_done()
                break

            async with visited_lock:
                if url in visited:
                    queue.task_done()
                    continue
                visited.add(url)

            try:
                # RACE CONDITION: Navigate OR Stop
                # We create a task for navigation so we can cancel it if stop is pressed first
                nav_task = asyncio.create_task(page.goto(url, wait_until="domcontentloaded", timeout=20000))
                stop_wait_task = asyncio.create_task(stop_event.wait())

                done, pending = await asyncio.wait([nav_task, stop_wait_task], return_when=asyncio.FIRST_COMPLETED)

                if stop_event.is_set():
                    # Stop was pressed! Cancel navigation immediately.
                    nav_task.cancel()
                    try:
                        await nav_task
                    except asyncio.CancelledError:
                        pass
                    queue.task_done()
                    break

                # If we are here, navigation finished first
                stop_wait_task.cancel()
                await nav_task # Propagate exceptions if any

                # Canonical URL (Handle Redirects e.g., Booking.com)
                current_url = page.url
                
                # Check stop again before heavy processing
                if stop_event.is_set():
                    queue.task_done()
                    break
                
                # Cleanup
                await self._close_popups(page)
                await self._handle_captcha(page)

                title = await page.title()
                content_html = await page.content()
                
                # Check stop again
                if stop_event.is_set():
                    queue.task_done()
                    break

                # Deduplication & Extraction
                clean_content = self._clean_content(content_html)

                # Async persistence
                await db_queue.put((session_id, current_url, title, clean_content, depth, "success"))

                if depth < max_depth and not stop_event.is_set():
                    soup = BeautifulSoup(content_html, "html.parser")
                    discovered = []

                    for a in soup.find_all("a", href=True):
                        full = urljoin(current_url, a["href"]) # Use current_url for relative links
                        if full.startswith("http") and urlparse(full).netloc == urlparse(current_url).netloc:
                            if self.is_allowed(full, rp):
                                discovered.append(full)
                                self.allowed_links.append(full)
                            else:
                                self.blocked_links.append(full)

                    if depth < 2:
                        targets = list(set(discovered))[:50]
                    else:
                        async with visited_lock:
                            targets = self._get_best_links(list(set(discovered)), visited)

                    for t in targets:
                        await queue.put((t, depth + 1, max_depth))

            except Exception as e:
                # Don't log error if it was just a stop
                if not stop_event.is_set():
                     await db_queue.put((session_id, url, "Error", str(e), depth, "failed"))

            queue.task_done()

        await page.close()

    # ---------------- ENTRY ---------------- #

    async def crawl_url(self, url: str, save_folder: str = None, simulate: bool = False, recursive: bool = False, max_depth: int = 1, stop_event: asyncio.Event = None) -> dict:
        """
        Orchestrates the crawling process with enhanced features.
        """
        
        start_time = datetime.now()
        session_id = str(uuid.uuid4())
        
        # Reset stats
        self.allowed_links = []
        self.blocked_links = []

        # 1. URL Normalization
        url = url.strip()
        if not url.startswith("http"):
            url = "https://" + url

        visited = set()
        visited_lock = asyncio.Lock()

        queue = asyncio.Queue()
        db_queue = asyncio.Queue()

        if stop_event is None:
            stop_event = asyncio.Event()

        rp = self._get_robots_parser(url)
        # Note: We check robots on the normalized URL
        if not self.is_allowed(url, rp):
             self.blocked_links.append(url)
             return {"status": "blocked", "error": "Disallowed by robots.txt", "links": {"allowed": [], "blocked": [url]}}

        queue.put_nowait((url, 0, max_depth if recursive else 0))

        async with async_playwright() as p:

            # Optimized Launch Options
            launch_options = {
                "headless": not simulate,
                "args": ["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
            }
            
            browser = await p.chromium.launch(**launch_options)

            # High Concurrency unless simulating
            NUM = 15 if not simulate else 6 
            
            context_options = {}
            if not simulate:
                 # Add user agent to avoid basic blocks in headless
                 context_options["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

            contexts = [await browser.new_context(**context_options) for _ in range(NUM)]

            # Start DB writer
            db_task = asyncio.create_task(self._db_writer(db_queue))

            workers = [
                asyncio.create_task(
                    self._worker(i, queue, contexts[i], session_id,
                                 rp, visited, visited_lock, db_queue, simulate, stop_event)
                )
                for i in range(NUM)
            ]

            # Custom join to support stop_event
            # We wait for queue to be empty OR stop_event to be set
            while not queue.empty() or (not stop_event.is_set() and len(visited) == 0): # Condition to wait needs to be robust
                 if stop_event.is_set():
                     break
                 if queue.empty() and queue._unfinished_tasks == 0:
                     break
                 await asyncio.sleep(0.5)
            
            # If we are here, either queue is empty (done) or stopped
            if not stop_event.is_set():
                await queue.join()

            # Shutdown signals
            for _ in workers:
                await queue.put(None)

            await asyncio.gather(*workers)

            await db_queue.join()
            await db_queue.put(None)
            await db_task

            rows = get_all_pages(session_id)

            full = ""
            for r in rows:
                full += f"\n\n== {r['title']} ==\n{r['content']}"

            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Prepare data
            rows_data = [dict(r) for r in rows] 
            
            # File Saving Logic
            saved_files = []
            if save_folder: # Fixed: Save even if stopped!
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                report_path = os.path.join(save_folder, "crawl_report.json")
                report_data = {
                    "url": url,
                    "session_id": session_id,
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": duration,
                    "pages_crawled": len(rows),
                    "pages": rows_data,
                    "links": {
                        "allowed_sample": self.allowed_links[:100], 
                        "blocked_sample": self.blocked_links[:100],
                        "total_allowed": len(self.allowed_links),
                        "total_blocked": len(self.blocked_links)
                    }
                }
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2)
                saved_files.append(report_path)

                content_path = os.path.join(save_folder, "crawled_content.txt")
                with open(content_path, "w", encoding="utf-8") as f:
                    f.write(full)
                saved_files.append(content_path)

            status = "stopped" if stop_event.is_set() else "success"

            return {
                "url": url,
                "status": status,
                "session_id": session_id,
                "pages_crawled": len(rows),
                "full_text": full,
                "content_preview": full[:500] + "...",
                "links": {"allowed": list(set(self.allowed_links)), "blocked": list(set(self.blocked_links))},
                "duration": round(duration, 2),
                "crawl_only_duration": round(duration, 2),
                "saved_files": saved_files,
                "database_records": rows_data
            }
