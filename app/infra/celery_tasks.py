"""
Celery Tasks
"""
import os
import asyncio
import logging
import time

from app.infra.celery_app import celery_app
from app.infra.job_tracker import JobTracker

logger = logging.getLogger(__name__)


@celery_app.task(name="ingestion.run_files")
def run_ingestion_files(job_id: str, file_paths: list, metadatas: list, reset_db: bool, tenant_id: str = None):
    """Run file ingestion in a Celery worker."""
    tracker = JobTracker(job_id, {
        "status": "pending",
        "chunks_added": 0,
        "total_chunks": 0,
        "job_id": job_id,
        "logs": ["Celery task started for file ingestion."],
    })
    try:
        t0 = time.perf_counter()
        from app.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(tenant_id=tenant_id)
        pipeline.run_ingestion(
            file_paths=file_paths,
            metadatas=metadatas,
            reset_db=reset_db,
            job_tracker=tracker,
        )
        tracker["status"] = "completed"
        tracker["ingestion_seconds"] = round(time.perf_counter() - t0, 2)
        tracker["logs"].append("Celery ingestion completed.")
    except Exception as e:
        tracker["status"] = "failed"
        tracker["error"] = str(e)
        tracker["logs"].append(f"Celery ingestion failed: {e}")
        logger.error(f"[CELERY] File ingestion failed: {e}")


@celery_app.task(name="ingestion.run_crawler")
def run_crawler_job(job_id: str, url: str, max_depth: int, save_folder: str, mode: str, tenant_id: str = None):
    """Run crawler + ingestion in a Celery worker."""
    tracker = JobTracker(job_id, {
        "status": "pending",
        "chunks_added": 0,
        "total_chunks": 0,
        "job_id": job_id,
        "logs": [f"Celery crawler started for URL: {url}"],
    })
    try:
        import sys
        from urllib.parse import urlparse

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        from app.ingestion.crawler_service import CrawlerService
        from app.ingestion.pipeline import IngestionPipeline

        crawler = CrawlerService(tenant_id=tenant_id)
        pipeline = IngestionPipeline(tenant_id=tenant_id)
        reset_db = mode == "overwrite"
        is_first_batch = [True]

        async def process_live_batch(batch_items):
            tracker["status"] = "crawling_and_extracting"
            tracker["logs"].append(f"Streaming {len(batch_items)} scraped pages to Vector Engine...")

            import hashlib

            paths = []
            metas = []
            target_domain = urlparse(url).netloc
            domain_folder = os.path.join(save_folder, target_domain)
            os.makedirs(domain_folder, exist_ok=True)

            for item in batch_items:
                current_url = item[1]
                title = item[2]
                content = item[3]

                safe_name = hashlib.md5(current_url.encode()).hexdigest()
                path = os.path.join(domain_folder, safe_name + ".txt")

                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"== {title} ==\n{content}")

                paths.append(path)
                metas.append(
                    {
                        "type": "url",
                        "source_url": current_url,
                        "source_domain": urlparse(current_url).netloc,
                        "document_type": "webpage",
                    }
                )

            reset = reset_db if is_first_batch[0] else False
            is_first_batch[0] = False

            def ingest_sync():
                t_ingest = time.perf_counter()
                pipeline.run_ingestion(
                    paths,
                    metadatas=metas,
                    reset_db=reset,
                    job_tracker=tracker,
                    mark_completed=False,
                )
                elapsed = time.perf_counter() - t_ingest
                tracker.setdefault("logs", []).append(
                    f"Batch ingestion completed in {elapsed:.2f}s for {len(paths)} files."
                )

            await asyncio.to_thread(ingest_sync)

        target_domain = urlparse(url).netloc
        domain_folder = os.path.join(save_folder, target_domain)

        t_crawl = time.perf_counter()
        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url,
                save_folder=domain_folder,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
                on_batch_extracted=process_live_batch,
            )
        )
        crawl_elapsed = time.perf_counter() - t_crawl
        tracker["crawl_seconds"] = round(crawl_elapsed, 2)
        loop.close()

        if result.get("saved_files") or not is_first_batch[0]:
            tracker["status"] = "completed"
            tracker["logs"].append(f"Celery crawler completed in {crawl_elapsed:.2f}s.")
        else:
            tracker["status"] = "failed"
            tracker["error"] = "No unstructured text output generated by crawler."
    except Exception as e:
        tracker["status"] = "failed"
        tracker["error"] = str(e)
        tracker["logs"].append(f"Celery crawler failed: {e}")
        logger.error(f"[CELERY] Crawler failed: {e}")


# -----------------------------------------------------------------------------
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
        from app.infra.database import upsert_ingestion_job
        from app.v2.retrieval.pageindex_tool import store_documents_in_tree_cache
        import asyncio
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from app.v2.ingestion.crawler_v2 import CrawlerService as CrawlerServiceV2
        crawler = CrawlerServiceV2(tenant_id=tenant_id)
        
        from urllib.parse import urlparse as _urlparse
        _domain = _urlparse(url).netloc or "unknown"
        _t = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (tenant_id or "global"))[:64] or "global"
        _save_folder = os.path.join("data", "crawled_docs_v2", _t, _domain)
        os.makedirs(_save_folder, exist_ok=True)

        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url,
                save_folder=_save_folder,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
            )
        )
        loop.close()
        docs = [
            {"filename": r.get("url", "page"), "content": r.get("content", "")}
            for r in crawler.results_memory
            if r.get("status") == "success"
        ]
        node_count = store_documents_in_tree_cache(session_id=session_id, documents=docs, tenant_id=tenant_id)
        tracker["status"] = "completed"
        tracker["logs"].append(f"V2 Crawler extracted {len(docs)} pages into {node_count} nodes.")
        upsert_ingestion_job(
            job_id=job_id,
            status="completed",
            payload={
                "message": f"V2 crawler indexed {len(docs)} pages into {node_count} nodes.",
                "session_id": session_id,
                "nodes_indexed": node_count,
            },
            tenant_id=tenant_id,
        )
    except Exception as e:
        tracker["status"] = "failed"
        tracker["error"] = str(e)
        try:
            from app.infra.database import upsert_ingestion_job
            upsert_ingestion_job(
                job_id=job_id,
                status="failed",
                payload={"message": str(e), "session_id": session_id},
                tenant_id=tenant_id,
            )
        except Exception:
            pass
        logger.error(f"[CELERY V2] Crawler failed: {e}")
