import sys
sys.path.insert(0, r'D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC')
from app.multimodal.multimodal_router import MultimodalRouter
from app.supervisor.router import ExecutionGraph
import asyncio

router = MultimodalRouter()
info = router.ingest_files_for_session(
    question='Check the attached resume and recommend best Learnnect courses for this person.',
    files=[('Updated_Resume_DS.pdf', open('test-files/Updated_Resume_DS.pdf','rb').read())],
    session_id='debug-mm-001',
    image_mode='auto'
)
print('ingest', info)

async def main():
    graph = ExecutionGraph()
    res = await graph.invoke(
        query='Check the attached resume and recommend best Learnnect courses for this person.',
        session_id='debug-mm-001',
        model_provider='auto',
        extra_collections=[info['collection_name']],
        reranker_profile='llm_judge',
        reranker_model_name='llama-3.1-8b-instant',
    )
    print('answer', res.get('answer','')[:300])

asyncio.run(main())
