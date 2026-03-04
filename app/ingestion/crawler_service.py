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
import uuid
import math
import random
import hashlib
import time
from typing import Optional, Dict
from collections import defaultdict
from urllib.parse import urlparse, urljoin, urldefrag, urlencode, parse_qs, urlunparse
from datetime import datetime

import aiohttp
from playwright.async_api import async_playwright, Page, BrowserContext
from selectolax.parser import HTMLParser
from app.infra.database import init_db, insert_page_async, get_all_pages, enable_wal
from app.infra.hardware import HardwareProbe
import logging

logger = logging.getLogger(__name__)

class CrawlerService:

    def __init__(self):
        init_db()
        enable_wal()
        self.allowed_links = []
        self.blocked_links = []
        self._proxy_pool = [
            p.strip() for p in os.getenv("CRAWLER_PROXY_URLS", "").split(",") if p.strip()
        ]
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._domain_semaphores = defaultdict(lambda: asyncio.Semaphore(int(os.getenv("CRAWLER_DOMAIN_CONCURRENCY", "6"))))
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._pattern_seen = set()
        self._content_hashes = set()

    # ---------------- SMART URL SCORING (NO KEYWORDS) ---------------- #

    def _url_entropy(self, url: str) -> float:
        probs = [url.count(c)/len(url) for c in set(url)]
        return -sum(p * math.log2(p) for p in probs)

    def _pattern_signature(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.lower()
        # Replace digits/hex/uuid-ish segments with placeholders
        parts = []
        for seg in path.split("/"):
            if not seg:
                continue
            if seg.isdigit():
                parts.append("{num}")
            elif len(seg) >= 8 and all(c in "0123456789abcdef" for c in seg.lower()):
                parts.append("{hex}")
            else:
                parts.append(seg)
        return "/".join(parts)

    def _get_best_links(self, links: list, visited: set, limit: int = 5) -> list:
        """
        Truly generic smart selection:
        - URL entropy
        - path length
        - depth
        - duplicate similarity
        """

        scored = []

        for link in links:
            u = urlparse(link)
            path = u.path

            depth = len(list(filter(None, path.split("/"))))
            entropy = self._url_entropy(link)
            plen = len(path)

            dup_penalty = 0
            signature = self._pattern_signature(link)
            if signature in self._pattern_seen:
                dup_penalty -= 5

            score = entropy + (2 <= depth <= 4) * 2 + (20 < plen < 120) * 2 + dup_penalty
            scored.append((score, link))

        scored.sort(reverse=True)
        best = [x[1] for x in scored[:limit]]
        for link in best:
            self._pattern_seen.add(self._pattern_signature(link))
        return best

    # ---------------- ROBOTS ---------------- #

    async def _get_robots_parser(self, url: str):
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base in self._robots_cache:
            return self._robots_cache[base]
        robots = f"{base}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            async with self._http_session.get(robots, timeout=5) as resp:
                if resp.status >= 400:
                    rp.parse([])
                else:
                    text = await resp.text(errors="ignore")
                    rp.parse(text.splitlines())
        except Exception:
            rp.parse([])
        self._robots_cache[base] = rp
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

    async def _auto_scroll(self, page: Page):
        steps = int(os.getenv("CRAWLER_SCROLL_STEPS", "6"))
        pause_ms = int(os.getenv("CRAWLER_SCROLL_PAUSE_MS", "350"))
        for _ in range(steps):
            try:
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await asyncio.sleep(pause_ms / 1000)
            except Exception:
                break

    def _clean_content(self, html: str) -> str:
        """Removes headers, footers, navs to extract main content."""
        tree = HTMLParser(html)
        for tag in tree.css("header,footer,nav,aside,script,style,noscript,iframe"):
            tag.decompose()
        use_markdown = os.getenv("CRAWLER_MARKDOWNIFY", "false").lower() == "true"
        if use_markdown:
            try:
                from markdownify import markdownify as md
                content = md(tree.html)
            except Exception:
                content = tree.body.text(separator="\n") if tree.body else tree.text(separator="\n")
        else:
            content = tree.body.text(separator="\n") if tree.body else tree.text(separator="\n")
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])
        return content

    def _should_fallback_to_browser(self, html: str) -> bool:
        if not html:
            return True
        # Heuristic for SPA shells
        tree = HTMLParser(html)
        low_text = len(tree.text(strip=True)) < int(os.getenv("CRAWLER_HTTP_MIN_CHARS", "200"))
        if low_text:
            selectors = [s.strip() for s in os.getenv("CRAWLER_WAIT_SELECTORS", "").split(",") if s.strip()]
            for sel in selectors:
                if tree.css_first(sel):
                    return False
        spa_markers = ["id=\"root\"", "id=\"app\"", "data-reactroot", "data-v-app"]
        if any(marker in html for marker in spa_markers) and low_text:
            return True
        return low_text

    async def _fetch_static(self, url: str, proxy: str = None) -> Optional[dict]:
        """Fast-path: attempt SSR HTML fetch via aiohttp before Playwright."""
        headers = {
            "User-Agent": os.getenv(
                "CRAWLER_HTTP_USER_AGENT",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        timeout = aiohttp.ClientTimeout(total=12)
        try:
            async with self._http_session.get(url, headers=headers, proxy=proxy, timeout=timeout) as resp:
                if resp.status >= 400:
                    return None
                html = await resp.text(errors="ignore")
                if self._should_fallback_to_browser(html):
                    return None
                tree = HTMLParser(html)
                title = tree.css_first("title").text().strip() if tree.css_first("title") else url
                return {"url": url, "title": title, "html": html}
        except Exception:
            return None

    # ---------------- ASYNC DB WRITER ---------------- #

    async def _db_writer(self, queue, on_batch_extracted=None):
        """
        Dedicated persistence coroutine.
        Workers push records here.
        This completely removes SQLite from crawler critical path.
        """
        batch = []
        while True:
            item = await queue.get()
            if item is None:
                if batch and on_batch_extracted:
                    await on_batch_extracted(list(batch))
                break
                
            insert_page_async(*item)
            
            # Execute batch callback hook explicitly for streaming pipeline architecture
            if item[5] == "success" and on_batch_extracted:
                batch.append(item)
                if len(batch) >= 5:
                    await on_batch_extracted(list(batch))
                    batch.clear()
                    
            queue.task_done()

    # ---------------- WORKER ---------------- #

    async def _worker(self, wid, queue, context, session_id, rp,
                      visited, visited_lock, db_queue, stop_event, max_urls: int, start_ts: float, max_seconds: float, http_only: bool):

        # Always block heavy resources for speed.
        try:
            async def _route_handler(route):
                if route.request.resource_type in {"image", "font", "media"}:
                    await route.abort()
                else:
                    await route.continue_()
            await context.route("**/*", _route_handler)
        except Exception:
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
            if max_seconds and (time.perf_counter() - start_ts) > max_seconds:
                stop_event.set()
            if max_urls and len(visited) >= max_urls:
                stop_event.set()

            # Check if stopped mid-work
            if stop_event.is_set():
                queue.task_done()
                break

            async with visited_lock:
                def normalize_url(u):
                    parsed = urlparse(u)
                    path = parsed.path.rstrip("/") or "/"
                    qs = parse_qs(parsed.query)
                    for bad in ["session", "ref", "uid", "token", "source", "click"]:
                        qs.pop(bad, None)
                    query = urlencode(qs, doseq=True)
                    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.params, query, ""))

                norm_url = normalize_url(url)
                if norm_url in visited:
                    queue.task_done()
                    continue
                visited.add(norm_url)

            try:
                # RACE CONDITION: Navigate OR Stop
                # We create a task for navigation so we can cancel it if stop is pressed first
                use_fast_http = os.getenv("CRAWLER_FAST_HTTP", "true").lower() == "true"
                proxy = None
                if self._proxy_pool:
                    proxy = random.choice(self._proxy_pool)

                if use_fast_http:
                    async with self._domain_semaphores[urlparse(url).netloc]:
                        fast = await self._fetch_static(url, proxy=proxy)
                    if fast:
                        content_html = fast["html"]
                        title = fast["title"]
                        current_url = fast["url"]
                        clean_content = self._clean_content(content_html)
                        content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
                        if content_hash in self._content_hashes:
                            queue.task_done()
                            continue
                        self._content_hashes.add(content_hash)
                        await db_queue.put((session_id, current_url, title, clean_content, depth, "success"))
                        if depth < max_depth and not stop_event.is_set():
                            tree = HTMLParser(content_html)
                            discovered = []
                            for a in tree.css("a"): 
                                if not a.attributes.get("href"): continue 
                                a = {"href": a.attributes["href"]}
                                full = urljoin(current_url, a["href"])
                                full, _ = urldefrag(full)
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
                        queue.task_done()
                        continue
                    if http_only:
                        queue.task_done()
                        continue

                nav_task = None
                for attempt in range(3):
                    try:
                        async with self._domain_semaphores[urlparse(url).netloc]:
                            nav_task = asyncio.create_task(page.goto(url, wait_until="domcontentloaded", timeout=20000))
                        stop_wait_task = asyncio.create_task(stop_event.wait())
                        done, pending = await asyncio.wait([nav_task, stop_wait_task], return_when=asyncio.FIRST_COMPLETED)
                        if stop_event.is_set():
                            nav_task.cancel()
                            try:
                                await nav_task
                            except asyncio.CancelledError:
                                pass
                            queue.task_done()
                            break
                        stop_wait_task.cancel()
                        await nav_task
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)

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

                # If content looks empty, wait for network idle or selectors.
                clean_content = self._clean_content(content_html)
                content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
                if content_hash in self._content_hashes:
                    queue.task_done()
                    continue
                self._content_hashes.add(content_hash)
                min_chars = int(os.getenv("CRAWLER_HTTP_MIN_CHARS", "500"))
                if len(clean_content) < min_chars:
                    selectors = [s.strip() for s in os.getenv("CRAWLER_WAIT_SELECTORS", "").split(",") if s.strip()]
                    try:
                        if selectors:
                            for sel in selectors:
                                await page.wait_for_selector(sel, timeout=3000)
                                break
                        else:
                            await page.wait_for_load_state("networkidle", timeout=3000)
                    except Exception:
                        pass
                    if os.getenv("CRAWLER_SCROLL", "true").lower() == "true":
                        await self._auto_scroll(page)
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
                    tree = HTMLParser(content_html)
                    discovered = []

                    for node in tree.css("a"):
                        if not node.attributes.get("href"): continue
                        a = {"href": node.attributes["href"]}
                        full = urljoin(current_url, a["href"]) # Use current_url for relative links
                        full, _ = urldefrag(full) # Remove URL HTML anchor fragments to prevent infinite loop duplication
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

    async def crawl_url(self, url: str, save_folder: str = None, simulate: bool = False, recursive: bool = False, max_depth: int = 1, stop_event: asyncio.Event = None, on_batch_extracted=None) -> dict:
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
        self._pattern_seen = set()
        self._content_hashes = set()

        if stop_event is None:
            stop_event = asyncio.Event()

        self._http_session = aiohttp.ClientSession()
        rp = await self._get_robots_parser(url)
        # Note: We check robots on the normalized URL
        if not self.is_allowed(url, rp):
             self.blocked_links.append(url)
             return {"status": "blocked", "error": "Disallowed by robots.txt", "links": {"allowed": [], "blocked": [url]}}

        queue.put_nowait((url, 0, max_depth if recursive else 0))
        max_urls = int(os.getenv("CRAWLER_MAX_URLS", "5000"))
        max_seconds = float(os.getenv("CRAWLER_MAX_SECONDS", "0"))
        start_ts = time.perf_counter()
        http_only = os.getenv("CRAWLER_FAST_HTTP_ONLY", "false").lower() == "true"

        async with async_playwright() as p:

            # Optimized Launch Options
            launch_options = {
                "headless": True,
                "args": ["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
            }
            
            browser = await p.chromium.launch(**launch_options)

            # High Concurrency unless simulating (Phase 13 Dynamic Hardware Hook)
            profile = HardwareProbe.get_profile()
            NUM = profile.get("crawler_workers", 4)
            logger.info(f"[CRAWLER HARDWARE SCALE] Instantiating {NUM} parallel headless Chromium workers.")
            
            context_options = {}
            # Add user agent to avoid basic blocks in headless
            context_options["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            if self._proxy_pool:
                # Playwright expects a single proxy per context; rotate per context.
                context_options["proxy"] = {"server": random.choice(self._proxy_pool)}

            contexts = [await browser.new_context(**context_options) for _ in range(NUM)]

            # Start DB writer
            db_task = asyncio.create_task(self._db_writer(db_queue, on_batch_extracted))

            workers = [
                asyncio.create_task(
                    self._worker(i, queue, contexts[i], session_id,
                                 rp, visited, visited_lock, db_queue, stop_event, max_urls, start_ts, max_seconds, http_only)
                )
                for i in range(NUM)
            ]

            # Wait for completion or stop
            if not stop_event.is_set():
                await queue.join()

            # Shutdown signals
            for _ in workers:
                await queue.put(None)

            await asyncio.gather(*workers)

            await db_queue.join()
            await db_queue.put(None)
            await db_task
            if self._http_session:
                await self._http_session.close()
                self._http_session = None

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
                
                report_path = os.path.join(save_folder, "metadata.json")
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
                import json
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2)
                saved_files.append(report_path)

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
