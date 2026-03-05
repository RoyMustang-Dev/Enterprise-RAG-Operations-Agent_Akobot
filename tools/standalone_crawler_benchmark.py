#!/usr/bin/env python3
"""
Standalone crawler benchmark (no repo dependencies).
Usage:
  python standalone_crawler_benchmark.py --url https://example.com --depth 3
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import time
import urllib.robotparser
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict
from urllib.parse import urlparse, urljoin, urldefrag, urlencode, parse_qs, urlunparse
from xml.etree import ElementTree

import aiohttp
from selectolax.parser import HTMLParser
from playwright.async_api import async_playwright, Page


class SeenUrlStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY, domain TEXT, ts TEXT)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_domain ON seen_urls(domain)")
            conn.commit()

    def get_all(self) -> set:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT url FROM seen_urls").fetchall()
            return {r[0] for r in rows}

    def record(self, url: str, domain: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen_urls(url, domain, ts) VALUES (?, ?, ?)",
                (url, domain, datetime.utcnow().isoformat()),
            )
            conn.commit()


class CrawlerService:
    def __init__(self, out_dir: str, seen_db: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self._seen_store = SeenUrlStore(seen_db)
        self.allowed_links = []
        self.blocked_links = []
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._domain_semaphores = defaultdict(lambda: asyncio.Semaphore(6))
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._pattern_seen = set()
        self._content_hash_counts = defaultdict(int)
        self._content_hash_cap = 2

    def _canonicalize_url(self, url: str) -> str:
        if not url:
            return url
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        if netloc.endswith(":443"):
            netloc = netloc[:-4]
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        qs = parse_qs(parsed.query, keep_blank_values=False)
        for bad in [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "ref", "source", "session", "uid", "token", "click",
            "sort", "order", "dir", "page", "p", "per_page", "perpage", "view", "layout", "grid",
            "filter", "filters", "facet", "facets", "brand", "size", "color", "variant", "price",
            "min", "max",
            "checkin", "checkout", "date", "from", "to", "guests", "rooms", "adults", "children",
        ]:
            qs.pop(bad, None)
        query = urlencode(qs, doseq=True)
        return urlunparse((scheme, netloc, path, parsed.params, query, ""))

    def _extract_canonical(self, html: str, fallback_url: str) -> str:
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
        domain = (domain or "").lower()
        keywords = [
            "shop", "store", "cart", "checkout", "booking", "hotel", "flight",
            "travel", "restaurant", "ticket", "reserve", "order",
        ]
        return any(k in domain for k in keywords)

    def _force_playwright_domain(self, domain: str) -> bool:
        domain = (domain or "").lower()
        spa_domains = [
            "notion.so",
            "www.notion.so",
        ]
        return domain in spa_domains

    def _is_allowed_path_for_restrictive(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        allowed = ["/product", "/category", "/docs", "/blog", "/help", "/support"]
        return any(path.startswith(p) or f"/{p.strip('/')}/" in path for p in allowed)

    def _pagination_over_cap(self, url: str, cap: int = 3) -> bool:
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            for key in ["page", "p"]:
                if key in qs:
                    if int(qs[key][0]) > cap:
                        return True
            parts = [p for p in parsed.path.split("/") if p]
            for i, part in enumerate(parts[:-1]):
                if part == "page" and int(parts[i + 1]) > cap:
                    return True
        except Exception:
            return False
        return False

    async def _fetch_sitemap_urls(self, sitemap_url: str, domain: str, limit: int = 500) -> list:
        urls = []
        try:
            async with self._http_session.get(sitemap_url, timeout=8) as resp:
                if resp.status >= 400:
                    return urls
                max_bytes = 2_000_000
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

    def _url_entropy(self, url: str) -> float:
        probs = [url.count(c) / len(url) for c in set(url)]
        return -sum(p * math.log2(p) for p in probs)

    def _pattern_signature(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.lower()
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
        scored = []
        for link in links:
            u = urlparse(link)
            path = u.path
            depth = len([p for p in path.split("/") if p])
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
            return rp.can_fetch("*", url)
        except Exception:
            return True

    async def _auto_scroll(self, page: Page):
        try:
            dims = await page.evaluate("({h: document.body.scrollHeight, v: window.innerHeight})")
            total = int(dims.get("h", 0))
            view = int(dims.get("v", 0)) or 1
            steps = max(1, min(6, int(total / view)))
        except Exception:
            steps = 2
        for _ in range(steps):
            try:
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await asyncio.sleep(0.25)
            except Exception:
                break

    def _clean_content(self, html: str) -> str:
        tree = HTMLParser(html)
        for tag in tree.css("header,footer,nav,aside,script,style,noscript,iframe"):
            tag.decompose()
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
        tree = HTMLParser(html)
        text_len = len(tree.text(strip=True))
        html_len = len(html)
        low_text_ratio = (text_len / max(1, html_len)) < 0.01
        low_text_abs = text_len < 200
        spa_markers = ["id=\"root\"", "id=\"app\"", "data-reactroot", "data-v-app"]
        if any(marker in html for marker in spa_markers) and (low_text_ratio or low_text_abs):
            return True
        return low_text_ratio or low_text_abs

    async def _fetch_static(self, url: str) -> Optional[dict]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        timeout = aiohttp.ClientTimeout(total=12)
        try:
            async with self._http_session.get(url, headers=headers, timeout=timeout) as resp:
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

    async def _fetch_playwright(self, url: str) -> Optional[dict]:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await self._auto_scroll(page)
                html = await page.content()
                title = await page.title()
                await context.close()
                await browser.close()
                return {"url": url, "title": title or url, "html": html}
        except Exception:
            return None

    async def crawl_url(
        self,
        url: str,
        max_depth: int = 3,
        max_urls: int = 200,
        max_seconds: float = 30,
        http_only: bool = True,
    ) -> dict:
        self._http_session = aiohttp.ClientSession()
        seed = self._canonicalize_url(url)
        rp = await self._get_robots_parser(seed)
        domain = urlparse(seed).netloc

        seed_urls = [seed]
        sitemaps = await self._discover_sitemaps(seed, rp)
        for sm in sitemaps:
            seed_urls.extend(await self._fetch_sitemap_urls(sm, domain))
        seed_urls = list(dict.fromkeys(seed_urls))[:200]

        queue = asyncio.Queue()
        visited = set()
        seen = self._seen_store.get_all()
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
            if not norm_url or norm_url in visited or norm_url in seen:
                queue.task_done()
                continue
            if not self.is_allowed(norm_url, rp):
                self.blocked_links.append(norm_url)
                queue.task_done()
                continue

            visited.add(norm_url)
            self._seen_store.record(norm_url, urlparse(norm_url).netloc)

            content = await self._fetch_static(norm_url)
            if (not content) and (not http_only or self._force_playwright_domain(domain)):
                content = await self._fetch_playwright(norm_url)
            if not content:
                queue.task_done()
                continue

            content_html = content["html"]
            title = content["title"]
            current_url = self._extract_canonical(content_html, content["url"])
            clean_content, quality_score, is_thin = self._prepare_content(content_html)
            if is_thin:
                queue.task_done()
                continue

            content_hash = hashlib.sha256(clean_content.encode("utf-8", errors="ignore")).hexdigest()
            if self._content_hash_counts[content_hash] >= self._content_hash_cap:
                queue.task_done()
                continue
            self._content_hash_counts[content_hash] += 1

            doc = {
                "url": current_url,
                "title": title,
                "content": clean_content,
                "depth": depth,
                "quality_score": quality_score,
                "ts": datetime.utcnow().isoformat(),
            }
            out_path = os.path.join(self.out_dir, f"{hashlib.md5(current_url.encode()).hexdigest()}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            pages_crawled += 1

            if depth < max_depth:
                tree = HTMLParser(content_html)
                discovered = []
                for a in tree.css("a"):
                    href = a.attributes.get("href")
                    if not href:
                        continue
                    full = urljoin(current_url, href)
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
                best = self._get_best_links(targets, visited, limit=8)
                for t in best:
                    queue.put_nowait((t, depth + 1))

            queue.task_done()

        await self._http_session.close()
        return {
            "status": "success",
            "pages_crawled": pages_crawled,
            "links": {"allowed": self.allowed_links, "blocked": self.blocked_links},
        }


async def run_benchmark(args):
    crawler = CrawlerService(out_dir=args.out_dir, seen_db=args.seen_db)
    start = time.perf_counter()
    result = await crawler.crawl_url(
        url=args.url,
        max_depth=args.depth,
        max_urls=args.max_urls,
        max_seconds=args.max_seconds,
        http_only=args.http_only,
    )
    end = time.perf_counter()
    total = end - start
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)
    print(f"Total Time Taken : {total:.2f} seconds")
    print(f"Pages Crawled    : {result.get('pages_crawled', 0)}")
    print(f"Allowed Links    : {len(result.get('links', {}).get('allowed', []))}")
    print(f"Status           : {result.get('status')}")
    print(f"Output Folder    : {os.path.abspath(args.out_dir)}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone crawler benchmark")
    p.add_argument("--url", required=True, help="Seed URL to crawl")
    p.add_argument("--depth", type=int, default=3, help="Max crawl depth")
    p.add_argument("--max-urls", type=int, default=200, help="Max URLs to crawl")
    p.add_argument("--max-seconds", type=float, default=30, help="Time limit in seconds")
    p.add_argument("--http-only", action="store_true", help="HTTP-only (no Playwright fallback)")
    p.add_argument("--out-dir", default="./crawler_out", help="Output folder for crawled JSON")
    p.add_argument("--seen-db", default="./crawler_out/seen_urls.db", help="SQLite DB path for seen URLs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_benchmark(args))
