import asyncio
from app.ingestion.crawler_service import CrawlerService

async def main():
    crawler = CrawlerService()
    print("Executing Crawl...")
    try:
        await crawler.crawl_url("https://example.com", recursive=False)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
