from qdrant_client import QdrantClient

client = QdrantClient(path="data/qdrant_storage")
try:
    count = client.get_collection("enterprise_rag").points_count
    print(f"Total points in enterprise_rag: {count}")
    
    # Let's also retrieve one point just to see the metadata
    scroll_res = client.scroll("enterprise_rag", limit=1)
    if scroll_res[0]:
        print("Sample payload:", scroll_res[0][0].payload)
except Exception as e:
    print("Error:", e)
