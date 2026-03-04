import sys
import asyncio

sys.path.insert(0, r'D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC')
from app.supervisor.router import ExecutionGraph

async def main():
    graph = ExecutionGraph()
    try:
        res = await graph.invoke(
            query='Check the attached resume and recommend best Learnnect courses for this person.',
            session_id='debug-local-001',
            model_provider='auto',
            extra_collections=[],
            reranker_profile='llm_judge',
            reranker_model_name='llama-3.1-8b-instant',
        )
        print('OK', res.get('answer','')[:200])
    except Exception as e:
        import traceback
        print('ERROR', e)
        print(traceback.format_exc())

asyncio.run(main())
