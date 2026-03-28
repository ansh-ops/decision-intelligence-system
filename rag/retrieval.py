from rag.memory_store import FAISSAnalysisMemory


memory = FAISSAnalysisMemory()


def retrieve_similar_analyses(query_text: str, top_k: int = 3):
    return memory.query(query_text, top_k=top_k)


def store_analysis_summary(summary_text: str, metadata: dict):
    memory.add(summary_text, metadata)