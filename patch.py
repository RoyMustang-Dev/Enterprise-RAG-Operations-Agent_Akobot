import re

with open('app/v2/ingestion/crawler_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('async def _db_writer(self, queue, on_batch_extracted=None):', 'async def _memory_writer(self, queue, on_batch_extracted=None):')
content = content.replace('insert_page_async(*item, tenant_id=self.tenant_id)', 'self.results_memory.append({"session_id": item[0], "url": item[1], "title": item[2], "content": item[3], "depth": item[4], "status": item[5]})')
content = content.replace('db_task = asyncio.create_task(self._db_writer(db_queue, on_batch_extracted))', 'db_task = asyncio.create_task(self._memory_writer(db_queue, on_batch_extracted))')

# The big get_all_pages replacement
old_rows = r"rows = get_all_pages(session_id, tenant_id=self.tenant_id)"
new_rows = r"rows = [r for r in self.results_memory if r['session_id'] == session_id and r['status'] == 'success']"
content = content.replace(old_rows, new_rows)

old_data = r"rows_data = [dict(r) for r in rows]"
new_data = r"rows_data = rows"
content = content.replace(old_data, new_data)

with open('app/v2/ingestion/crawler_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
