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
from xml.etree import ElementTree

import aiohttp
from playwright.async_api import async_playwright, Page, BrowserContext
from selectolax.parser import HTMLParser
from app.infra.database import init_db, insert_page_async, get_all_pages, enable_wal, init_seen_urls_db, get_seen_urls, record_seen_url
from app.infra.hardware import HardwareProbe
import logging

logger = logging.getLogger(__name__)

class CrawlerService:

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        init_db(tenant_id=tenant_id)
        enable_wal(tenant_id=tenant_id, db_kind="crawler")
        init_seen_urls_db(tenant_id=tenant_id)
        self.allowed_links = []
        self.blocked_links = []
        self._proxy_pool = [
            p.strip() for p in os.getenv("CRAWLER_PROXY_URLS", "").split(",") if p.strip()
        ]
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        # Dynamic concurrency based on hardware profile (no .env knobs)
        profile = HardwareProbe.get_profile()
        workers = int(profile.get("crawler_workers", 6))
        self._domain_semaphores = defaultdict(lambda: asyncio.Semaphore(max(2, min(8, workers))))
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._pattern_seen = set()
        self._content_hashes = set()
        self._content_hash_counts = defaultdict(int)
        self._content_hash_cap = 2

    def _canonicalize_url(self, url: str) -> str:
        """Normalize URLs for de-duplication: strip fragments, tracking params, normalize host/path."""
        if not url:
            return url
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        # Remove default ports
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        if netloc.endswith(":443"):
            netloc = netloc[:-4]
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        # Strip common tracking params
        qs = parse_qs(parsed.query, keep_blank_values=False)
        for bad in [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "ref", "source", "session", "uid", "token", "click",
            # Faceted navigation + pagination + sorting
            "sort", "order", "dir", "page", "p", "per_page", "perpage", "view", "layout", "grid",
            "filter", "filters", "facet", "facets", "brand", "size", "color", "variant", "price",
            "min", "max",
            # Booking/search parameters
            "checkin", "checkout", "date", "from", "to", "guests", "rooms", "adults", "children",
        ]:
            qs.pop(bad, None)
        # Stable ordering
        query = urlencode(qs, doseq=True)
        return urlunparse((scheme, netloc, path, parsed.params, query, ""))

    def _extract_canonical(self, html: str, fallback_url: str) -> str:
        """If a canonical link exists, return it; otherwise return fallback_url."""
        if not html:
            return fallback_url
        try:
            tree = HTMLParser(html)
            link = tree.css_first("link[rel='canonical']")
            if link and link.attributes.get("href"):
                canon = self._canonicalize_url(link.attributes["href"])
                if canon:
                    return canon
        except Exception:
            pass
        return fallback_url

    def _is_restrictive_domain(self, domain: str) -> bool:
        """Heuristic: restrict crawling for ecommerce/booking-like domains to reduce crawl traps."""
        domain = (domain or "").lower()
        keywords = [
            "shop", "store", "cart", "checkout", "booking", "hotel", "flight",
            "travel", "restaurant", "ticket", "reserve", "order"
        ]
        return any(k in domain for k in keywords)

    def _force_playwright_domain(self, domain: str) -> bool:
        """Per-domain override to allow Playwright for SPA-heavy sites."""
        domain = (domain or "").lower()
        spa_domains = [
            "notion.so",
            "www.notion.so",
        ]
        return domain in spa_domains

    def _is_allowed_path_for_restrictive(self, url: str) -> bool:
        """Allowlist key content paths for restrictive domains."""
        path = urlparse(url).path.lower()
        allowed = ["/product", "/category", "/docs", "/blog", "/help", "/support"]
        return any(path.startswith(p) or f"/{p.strip('/')}/" in path for p in allowed)

    def _pagination_over_cap(self, url: str, cap: int = 3) -> bool:
        """Detect pagination and cap depth to avoid traps."""
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            for key in ["page", "p"]:
                if key in qs:
                    try:
                        if int(qs[key][0]) > cap:
                            return True
                    except Exception:
                        return False
            # Also detect /page/2 style
            path_parts = [p for p in parsed.path.split("/") if p]
            for i, part in enumerate(path_parts[:-1]):
                if part == "page":
                    try:
                        if int(path_parts[i + 1]) > cap:
                            return True
                    except Exception:
                        return False
        except Exception:
            return False
        return False

    async def _fetch_sitemap_urls(self, sitemap_url: str, domain: str, limit: int = 500) -> list:
        """Fetch and parse sitemap.xml for URL seeds."""
        urls = []
        try:
            async with self._http_session.get(sitemap_url, timeout=8) as resp:
                if resp.status >= 400:
                    return urls
                max_bytes = 2_000_000
                length = resp.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > max_bytes:
                    return urls
                raw = await resp.content.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    return urls
                text = raw.decode(errors="ignore")
            root = ElementTree.fromstring(text)
            for loc in root.findall(".//{*}loc"):
                if loc.text:
                    u = self._canonicalize_url(loc.text.strip())
                    if u and urlparse(u).netloc == domain:
                        urls.append(u)
                        if len(urls) >= limit:
                            break
        except Exception:
            return urls
        return urls

    async def _discover_sitemaps(self, base_url: str, rp: urllib.robotparser.RobotFileParser) -> list:
        """Extract sitemap URLs from robots.txt, with /sitemap.xml fallback."""
        sitemaps = []
        try:
            if hasattr(rp, "site_maps") and rp.site_maps():
                sitemaps.extend(rp.site_maps())
        except Exception:
            pass
        if not sitemaps:
            parsed = urlparse(base_url)
            sitemaps.append(f"{parsed.scheme}://{parsed.netloc}/sitemap.xml")
        return sitemaps[:3]

    async def _http_only_crawl(self, seed_urls: list, session_id: str, rp, max_depth: int, max_urls: int, max_seconds: float) -> dict:
        """HTTP-only crawl path (no Playwright) for maximum speed."""
        queue = asyncio.Queue()
        visited = set()
        start_ts = time.perf_counter()
        pages_crawled = 0

        for u in seed_urls:
            queue.put_nowait((u, 0))

        while not queue.empty():
            if max_seconds and (time.perf_counter() - start_ts) > max_seconds:
                break
            if max_urls and len(visited) >= max_urls:
                break

            url, depth = await queue.get()
            norm_url = self._canonicalize_url(url)
            if not norm_url or norm_url in visited:
                queue.task_done()
                continue
            if not self.is_allowed(norm_url, rp):
                self.blocked_links.append(norm_url)
                queue.task_done()
                continue

            visited.add(norm_url)
            try:
                domain = urlparse(norm_url).netloc
                record_seen_url(norm_url, domain, tenant_id=self.tenant_id)
            except Exception:
                pass

            fast = await self._fetch_static(norm_url)
            if not fast:
                queue.task_done()
                continue

            content_html = fast["html"]
            title = fast["title"]
            current_url = self._extract_canonical(content_html, fast["url"])
            clean_content, quality_score, is_thin = self._prepare_content(content_html)
            if is_thin:
                queue.task_done()
                continue
            content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
            if self._content_hash_counts[content_hash] >= self._content_hash_cap:
                queue.task_done()
                continue
            self._content_hashes.add(content_hash)
            self._content_hash_counts[content_hash] += 1
            insert_page_async(session_id, current_url, title, clean_content, depth, "success", tenant_id=self.tenant_id)
            pages_crawled += 1

            if depth < max_depth:
                tree = HTMLParser(content_html)
                discovered = []
                for a in tree.css("a"):
                    if not a.attributes.get("href"):
                        continue
                    full = urljoin(current_url, a.attributes["href"])
                    full = self._canonicalize_url(full)
                    if self._pagination_over_cap(full):
                        continue
                    if full.startswith("http") and urlparse(full).netloc == urlparse(current_url).netloc:
                        if self._is_restrictive_domain(urlparse(full).netloc) and not self._is_allowed_path_for_restrictive(full):
                            continue
                        if self.is_allowed(full, rp):
                            discovered.append(full)
                            self.allowed_links.append(full)
                        else:
                            self.blocked_links.append(full)
                targets = list(set(discovered))[:50]
                for t in targets:
                    queue.put_nowait((t, depth + 1))

            queue.task_done()

        return {
            "status": "success",
            "pages_crawled": pages_crawled,
            "links": {"allowed": self.allowed_links, "blocked": self.blocked_links},
        }

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
        # Auto-scroll only when content is likely longer than the viewport
        try:
            dims = await page.evaluate(
                "({h: document.body.scrollHeight, v: window.innerHeight})"
            )
            total = int(dims.get("h", 0))
            view = int(dims.get("v", 0)) or 1
            steps = max(1, min(6, int(total / view)))
        except Exception:
            steps = 2
        pause_ms = 250
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
        # Auto-switch to markdown if tables or lists are present
        use_markdown = bool(tree.css("table,ul,ol"))
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

    def _extract_headings(self, html: str) -> list:
        """Lightweight heading extraction (H1/H2/H3)."""
        if not html:
            return []
        try:
            tree = HTMLParser(html)
            headings = []
            for tag in ["h1", "h2", "h3"]:
                for node in tree.css(tag):
                    text = node.text(strip=True)
                    if text:
                        headings.append((tag.upper(), text[:160]))
            return headings
        except Exception:
            return []

    def _quality_score(self, text: str) -> float:
        """Fast content quality score (0-1) based on length + diversity."""
        if not text:
            return 0.0
        words = [w for w in text.split() if w]
        wc = len(words)
        if wc == 0:
            return 0.0
        unique_ratio = len(set(words)) / max(1, wc)
        length_score = min(1.0, wc / 200)
        return round((0.6 * length_score) + (0.4 * unique_ratio), 3)

    def _is_thin_content(self, text: str) -> bool:
        """Skip pages with ultra-low informational content."""
        if not text:
            return True
        wc = len(text.split())
        if wc < 40:
            return True
        lowered = text.lower()
        if wc < 80 and ("cookie" in lowered and "policy" in lowered):
            return True
        return False

    def _prepare_content(self, html: str) -> tuple:
        """Return (content_with_headings, quality_score, is_thin)."""
        clean = self._clean_content(html)
        headings = self._extract_headings(html)
        if headings:
            heading_block = "\n".join([f"{tag}: {text}" for tag, text in headings])
            clean = f"{heading_block}\n{clean}"
        score = self._quality_score(clean)
        thin = self._is_thin_content(clean)
        return clean, score, thin

    def _should_fallback_to_browser(self, html: str) -> bool:
        if not html:
            return True
        # Heuristic for SPA shells
        tree = HTMLParser(html)
        text_len = len(tree.text(strip=True))
        html_len = len(html)
        # If text is very sparse relative to HTML, assume SPA shell
        low_text_ratio = (text_len / max(1, html_len)) < 0.01
        low_text_abs = text_len < 200
        spa_markers = ["id=\"root\"", "id=\"app\"", "data-reactroot", "data-v-app"]
        if any(marker in html for marker in spa_markers) and (low_text_ratio or low_text_abs):
            return True
        return low_text_ratio or low_text_abs

    async def _fetch_static(self, url: str, proxy: str = None) -> Optional[dict]:
        """Fast-path: attempt SSR HTML fetch via aiohttp before Playwright."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
                
            insert_page_async(*item, tenant_id=self.tenant_id)
            
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
                norm_url = self._canonicalize_url(url)
                if norm_url in visited:
                    queue.task_done()
                    continue
                visited.add(norm_url)
                # Persist seen URL without blocking crawler
                try:
                    domain = urlparse(norm_url).netloc
                    asyncio.create_task(asyncio.to_thread(record_seen_url, norm_url, domain, self.tenant_id))
                except Exception:
                    pass

            try:
                # RACE CONDITION: Navigate OR Stop
                # We create a task for navigation so we can cancel it if stop is pressed first
                use_fast_http = os.getenv("CRAWLER_FAST_HTTP", "true").lower() == "true"
                proxy = None
                if self._proxy_pool:
                    proxy = random.choice(self._proxy_pool)

                fast = None
                if use_fast_http:
                    async with self._domain_semaphores[urlparse(url).netloc]:
                        fast = await self._fetch_static(url, proxy=proxy)
                if fast:
                    content_html = fast["html"]
                    title = fast["title"]
                    current_url = self._extract_canonical(content_html, fast["url"])
                    clean_content, quality_score, is_thin = self._prepare_content(content_html)
                    if is_thin:
                        queue.task_done()
                        continue
                    content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
                    if self._content_hash_counts[content_hash] >= self._content_hash_cap:
                        queue.task_done()
                        continue
                    self._content_hashes.add(content_hash)
                    self._content_hash_counts[content_hash] += 1
                    await db_queue.put((session_id, current_url, title, clean_content, depth, "success"))
                    if depth < max_depth and not stop_event.is_set():
                        tree = HTMLParser(content_html)
                        discovered = []
                        for a in tree.css("a"):
                            if not a.attributes.get("href"):
                                continue
                            a = {"href": a.attributes["href"]}
                            full = urljoin(current_url, a["href"])
                            full = self._canonicalize_url(full)
                            if self._pagination_over_cap(full):
                                continue
                            if full.startswith("http") and urlparse(full).netloc == urlparse(current_url).netloc:
                                if self._is_restrictive_domain(urlparse(full).netloc) and not self._is_allowed_path_for_restrictive(full):
                                    continue
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
                current_url = self._extract_canonical(content_html, current_url)

                # If content looks empty, wait for network idle or selectors.
                clean_content, quality_score, is_thin = self._prepare_content(content_html)
                content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
                if self._content_hash_counts[content_hash] >= self._content_hash_cap:
                    queue.task_done()
                    continue
                if is_thin:
                    queue.task_done()
                    continue
                self._content_hashes.add(content_hash)
                self._content_hash_counts[content_hash] += 1
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
                clean_content, quality_score, is_thin = self._prepare_content(content_html)
                if is_thin:
                    queue.task_done()
                    continue

                # Async persistence
                await db_queue.put((session_id, current_url, title, clean_content, depth, "success"))

                if depth < max_depth and not stop_event.is_set():
                    tree = HTMLParser(content_html)
                    discovered = []

                    for node in tree.css("a"):
                        if not node.attributes.get("href"): continue
                        a = {"href": node.attributes["href"]}
                        full = urljoin(current_url, a["href"]) # Use current_url for relative links
                        full = self._canonicalize_url(full) # Canonicalize and strip fragments/tracking
                        if self._pagination_over_cap(full):
                            continue
                        if full.startswith("http") and urlparse(full).netloc == urlparse(current_url).netloc:
                            if self._is_restrictive_domain(urlparse(full).netloc) and not self._is_allowed_path_for_restrictive(full):
                                continue
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
        url = self._canonicalize_url(url)

        visited = set()
        visited_lock = asyncio.Lock()

        queue = asyncio.Queue()
        db_queue = asyncio.Queue()
        self._pattern_seen = set()
        self._content_hashes = set()
        self._content_hash_counts = defaultdict(int)

        # Seed visited with previously seen URLs for this domain (persisted across runs)
        try:
            domain = urlparse(url).netloc
            visited.update(get_seen_urls(domain, tenant_id=self.tenant_id))
            # Ensure the seed URL is not pre-blocked by persisted history
            visited.discard(url)
        except Exception:
            pass

        if stop_event is None:
            stop_event = asyncio.Event()

        self._http_session = aiohttp.ClientSession()
        rp = await self._get_robots_parser(url)
        # Note: We check robots on the normalized URL
        if not self.is_allowed(url, rp):
             self.blocked_links.append(url)
             if self._http_session:
                 await self._http_session.close()
                 self._http_session = None
             return {"status": "blocked", "error": "Disallowed by robots.txt", "links": {"allowed": [], "blocked": [url]}}

        max_urls = int(os.getenv("CRAWLER_MAX_URLS", "5000"))
        max_seconds = float(os.getenv("CRAWLER_MAX_SECONDS", "0"))
        http_only = os.getenv("CRAWLER_FAST_HTTP_ONLY", "false").lower() == "true"
        if self._force_playwright_domain(urlparse(url).netloc):
            http_only = False
        elif http_only:
            # Dynamic SPA detection: if seed appears SPA-like, allow Playwright
            try:
                async with self._http_session.get(url, timeout=6) as resp:
                    if resp.status < 400:
                        html_probe = await resp.text(errors="ignore")
                        if self._should_fallback_to_browser(html_probe):
                            http_only = False
            except Exception:
                pass

        # Robots.txt + Sitemap bootstrap
        seed_urls = []
        try:
            domain = urlparse(url).netloc
            sitemap_urls = []
            for sm in await self._discover_sitemaps(url, rp):
                sitemap_urls.extend(await self._fetch_sitemap_urls(sm, domain))
            seed_urls = sitemap_urls[:200]
        except Exception:
            seed_urls = []
        if not seed_urls:
            seed_urls = [url]

        if http_only:
            result = await self._http_only_crawl(seed_urls, session_id, rp, max_depth if recursive else 0, max_urls, max_seconds)
            if self._http_session:
                await self._http_session.close()
                self._http_session = None
            return result

        for u in seed_urls:
            queue.put_nowait((u, 0, max_depth if recursive else 0))
        start_ts = time.perf_counter()

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

            rows = get_all_pages(session_id, tenant_id=self.tenant_id)

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
