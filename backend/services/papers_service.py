from __future__ import annotations

from functools import lru_cache

from backend.schemas.papers import PapersQueryResponse, PapersSource, PapersStatusResponse
from src.papers_rag import AcademicPapersRAG


@lru_cache(maxsize=1)
def get_rag_system() -> AcademicPapersRAG | None:
    rag = AcademicPapersRAG(
        papers_folder="papers/agents",
        chunk_size=512,
        chunk_overlap=50,
        similarity_top_k=5,
    )
    initialized = rag.initialize_system(force_rebuild=False)
    return rag if initialized else None


class PapersService:
    def __init__(self):
        self.papers_folder = "papers/agents"
        self.index_storage_path = "storage/papers_index"
        self.vector_db_path = "storage/papers_vectordb"

    def get_status(self) -> PapersStatusResponse:
        rag = get_rag_system()
        pdf_count = len(AcademicPapersRAG(papers_folder=self.papers_folder).list_indexed_papers())
        index_exists = AcademicPapersRAG()._index_exists()
        vector_store_exists = AcademicPapersRAG().vector_db_path.exists()
        ready = rag is not None

        return PapersStatusResponse(
            ready=ready,
            initialized=ready,
            papers_folder=self.papers_folder,
            pdf_count=pdf_count,
            index_exists=index_exists,
            vector_store_exists=vector_store_exists,
            message="" if ready else "The academic papers index is not ready.",
        )

    def query(self, query: str) -> PapersQueryResponse:
        rag = get_rag_system()
        if rag is None:
            return PapersQueryResponse(
                success=False,
                query=query,
                error="Failed to initialize the academic papers RAG system.",
            )

        result = rag.search_papers(query, include_metadata=True)
        return PapersQueryResponse(
            success=result.get("success", False),
            query=result.get("query", query),
            response=result.get("response", ""),
            sources=[PapersSource(**source) for source in result.get("sources", [])],
            search_time=result.get("search_time", 0),
            num_sources=result.get("num_sources", len(result.get("sources", []))),
            error=result.get("error", ""),
        )
