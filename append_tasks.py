with open('app/infra/celery_tasks.py', 'a', encoding='utf-8') as f:
    f.write('''\n\n# -----------------------------------------------------------------------------
# V2 Ingestion Tasks (PageIndex Tree Builders)
# -----------------------------------------------------------------------------
@celery_app.task(name="ingestion.run_files_v2")
def run_ingestion_files_v2(job_id: str, file_paths: list, tenant_id: str = None):
    tracker = JobTracker(job_id, {
        "status": "pending",
        "job_id": job_id,
        "logs": ["Celery V2 task started for PageIndex File extraction."],
    })
    try:
        import asyncio
        import sys
        
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from app.v2.ingestion.file_upload_v2 import FileUploadServiceV2
        # Mock UploadFile functionality for local paths
        class MockUploadFile:
            def __init__(self, path):
                self.filename = os.path.basename(path)
                self.path = path
            async def read(self):
                with open(self.path, "rb") as f:
                    return f.read()

        sv = FileUploadServiceV2(tenant_id=tenant_id)
        files = [MockUploadFile(p) for p in file_paths]
        
        result = loop.run_until_complete(sv.process_files(files))
        loop.close()
        
        tracker["status"] = "completed"
        tracker["logs"].append(f"V2 Extraction Complete: {len(result)} files processed for PageIndex.")
    except Exception as e:
        tracker["status"] = "failed"
        tracker["error"] = str(e)
        logger.error(f"[CELERY V2] File ingestion failed: {e}")

@celery_app.task(name="ingestion.run_crawler_v2")
def run_crawler_job_v2(job_id: str, url: str, max_depth: int, session_id: str, tenant_id: str = None):
    tracker = JobTracker(job_id, {
        "status": "pending",
        "job_id": job_id,
        "logs": [f"Celery V2 crawler started for URL: {url}"],
    })
    try:
        import asyncio
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from app.v2.ingestion.crawler_v2 import CrawlerServiceV2
        crawler = CrawlerServiceV2(tenant_id=tenant_id)
        
        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url,
                save_folder=None,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
                session_id=session_id
            )
        )
        loop.close()
        
        tracker["status"] = "completed"
        tracker["logs"].append(f"V2 Crawler extracted {len(result.get('saved_files', []))} nodes into memory.")
    except Exception as e:
        tracker["status"] = "failed"
        tracker["error"] = str(e)
        logger.error(f"[CELERY V2] Crawler failed: {e}")
''')
print("Appended V2 tasks!")
