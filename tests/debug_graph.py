import asyncio
from app.supervisor.router import ExecutionGraph

async def main():
    try:
        print("[DEBUG] Initializing Orchestrator...")
        orchestrator = ExecutionGraph()
        
        print("[DEBUG] Invoking ExecutionGraph natively...")
        result = await orchestrator.invoke("Hello", [])
        print("[DEBUG] Result:", result)
    except Exception as e:
        import traceback
        print("[FATAL ERROR TRACEBACK]")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
