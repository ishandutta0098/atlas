#!/usr/bin/env python3
"""
Academic Papers RAG System

This module implements a complete RAG (Retrieval-Augmented Generation) system
for academic research papers using LlamaIndex.

Features:
- PDF processing and text extraction
- Document chunking with overlap
- Vector embedding generation
- LanceDB vector storage
- Semantic search and retrieval
- Query engine with context
- Citation and reference tracking
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from src.pdf_processor import PaperPDFProcessor

load_dotenv()


class AcademicPapersRAG:
    """RAG system for academic research papers."""

    def __init__(
        self,
        papers_folder: str = "papers/agents",
        index_storage_path: str = "storage/papers_index",
        vector_db_path: str = "storage/papers_vectordb",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        similarity_top_k: int = 5,
    ):
        """
        Initialize the Academic Papers RAG system.

        Args:
            papers_folder (str): Path to folder containing PDF papers
            index_storage_path (str): Path to store the index
            vector_db_path (str): Path to store vector database
            chunk_size (int): Size of text chunks for indexing
            chunk_overlap (int): Overlap between chunks
            similarity_top_k (int): Number of similar chunks to retrieve
        """
        self.papers_folder = Path(papers_folder)
        self.index_storage_path = Path(index_storage_path)
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k

        # Initialize components
        self.pdf_processor = PaperPDFProcessor()
        self.index = None
        self.query_engine = None

        # Create storage directories
        self.index_storage_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure LlamaIndex settings
        self._configure_settings()

        # Initialize vector store
        self.vector_store = self._create_vector_store()

    def _configure_settings(self):
        """Configure LlamaIndex global settings."""
        from src.utils import get_config

        # Set up embeddings
        embedding_model = get_config(
            "api.openai.embedding_model", "text-embedding-3-small"
        )
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Set up LLM
        model = get_config("api.openai.model", "gpt-4o-mini")
        Settings.llm = OpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Set up node parser for chunking
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _create_vector_store(self):
        """Create and configure LanceDB vector store."""
        try:
            import lancedb

            # Connect to LanceDB
            db = lancedb.connect(str(self.vector_db_path))

            # Create vector store
            vector_store = LanceDBVectorStore(
                uri=str(self.vector_db_path), table_name="academic_papers"
            )

            return vector_store

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def load_papers(self) -> List[Document]:
        """
        Load and process all PDF papers from the papers folder.

        Returns:
            List[Document]: Processed documents
        """
        print(f"Loading papers from: {self.papers_folder}")

        if not self.papers_folder.exists():
            print(f"Papers folder does not exist: {self.papers_folder}")
            return []

        documents = self.pdf_processor.process_papers_folder(str(self.papers_folder))

        print(f"Loaded {len(documents)} documents")
        return documents

    def create_index(self, force_rebuild: bool = False) -> bool:
        """
        Create or load the vector index.

        Args:
            force_rebuild (bool): Force rebuild even if index exists

        Returns:
            bool: Success status
        """
        try:
            # Check if index already exists and we don't want to force rebuild
            if not force_rebuild and self._index_exists():
                print("Loading existing index...")
                self.index = self._load_existing_index()
                if self.index:
                    print("✓ Successfully loaded existing index")
                    return True

            print("Creating new index...")

            # Load papers
            documents = self.load_papers()

            if not documents:
                print("No documents to index")
                return False

            # Create storage context with vector store
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Create index
            print("Building vector index...")
            start_time = time.time()

            self.index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, show_progress=True
            )

            end_time = time.time()
            print(f"✓ Index created in {end_time - start_time:.2f} seconds")

            # Persist index
            print("Saving index to storage...")
            self.index.storage_context.persist(persist_dir=str(self.index_storage_path))
            print("✓ Index saved successfully")

            return True

        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def _index_exists(self) -> bool:
        """Check if an index already exists."""
        return (self.index_storage_path / "index_store.json").exists()

    def _load_existing_index(self):
        """Load existing index from storage."""
        try:
            # Recreate storage context with vector store
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_storage_path), vector_store=self.vector_store
            )

            # Load index
            index = load_index_from_storage(storage_context)
            return index

        except Exception as e:
            print(f"Error loading existing index: {e}")
            return None

    def setup_query_engine(self, temperature: float = 0.1) -> bool:
        """
        Setup the query engine for semantic search.

        Args:
            temperature (float): LLM temperature for response generation

        Returns:
            bool: Success status
        """
        if not self.index:
            print("Index not available. Please create index first.")
            return False

        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.similarity_top_k,
            )

            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
            )

            print("✓ Query engine setup successfully")
            return True

        except Exception as e:
            print(f"Error setting up query engine: {e}")
            return False

    def search_papers(
        self, query: str, include_metadata: bool = True
    ) -> Dict[str, any]:
        """
        Search for relevant papers based on the query.

        Args:
            query (str): Search query
            include_metadata (bool): Whether to include detailed metadata

        Returns:
            Dict[str, any]: Search results with response and sources
        """
        if not self.query_engine:
            return {
                "success": False,
                "error": "Query engine not initialized. Please setup query engine first.",
                "response": "",
                "sources": [],
            }

        try:
            print(f"Searching for: '{query}'")
            start_time = time.time()

            # Query the index
            response = self.query_engine.query(query)

            end_time = time.time()

            # Extract source information
            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_info = {
                        "text": (
                            node.text[:500] + "..."
                            if len(node.text) > 500
                            else node.text
                        ),
                        "score": getattr(node, "score", 0.0),
                    }

                    if include_metadata and hasattr(node, "metadata"):
                        metadata = node.metadata
                        source_info.update(
                            {
                                "file_name": metadata.get("file_name", "Unknown"),
                                "title": metadata.get("title", "Unknown Title"),
                                "authors": metadata.get("authors", "Unknown Authors"),
                                "page_count": metadata.get("page_count", 0),
                                "has_abstract": metadata.get("has_abstract", False),
                            }
                        )

                    sources.append(source_info)

            result = {
                "success": True,
                "response": str(response),
                "sources": sources,
                "query": query,
                "search_time": end_time - start_time,
                "num_sources": len(sources),
            }

            print(f"✓ Search completed in {end_time - start_time:.2f} seconds")
            print(f"Found {len(sources)} relevant sources")

            return result

        except Exception as e:
            print(f"Error during search: {e}")
            return {"success": False, "error": str(e), "response": "", "sources": []}

    def get_paper_summary(self, file_name: str) -> Dict[str, any]:
        """
        Get a summary of a specific paper.

        Args:
            file_name (str): Name of the PDF file

        Returns:
            Dict[str, any]: Paper summary information
        """
        # Find the paper in our processed documents
        papers_info = self.list_indexed_papers()

        target_paper = None
        for paper in papers_info:
            if paper["file_name"] == file_name:
                target_paper = paper
                break

        if not target_paper:
            return {
                "success": False,
                "error": f"Paper '{file_name}' not found in index",
            }

        # Generate summary using the RAG system
        summary_query = f"Provide a comprehensive summary of the paper '{target_paper['title']}', including its main contributions, methodology, and key findings."

        result = self.search_papers(summary_query, include_metadata=True)

        if result["success"]:
            return {
                "success": True,
                "paper_info": target_paper,
                "summary": result["response"],
                "sources_used": len(result["sources"]),
            }
        else:
            return result

    def list_indexed_papers(self) -> List[Dict[str, any]]:
        """
        List all papers that have been indexed.

        Returns:
            List[Dict[str, any]]: List of paper information
        """
        papers = []

        # Get papers from the folder
        if self.papers_folder.exists():
            pdf_files = list(self.papers_folder.glob("*.pdf"))

            for pdf_file in pdf_files:
                # Extract basic metadata
                metadata = self.pdf_processor.extract_pdf_metadata(str(pdf_file))

                paper_info = {
                    "file_name": pdf_file.stem,
                    "file_path": str(pdf_file),
                    "title": metadata.get("title", pdf_file.stem),
                    "authors": metadata.get("authors", "Unknown"),
                    "page_count": metadata.get("page_count", 0),
                    "file_size": pdf_file.stat().st_size if pdf_file.exists() else 0,
                }

                papers.append(paper_info)

        return papers

    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the complete RAG system.

        Args:
            force_rebuild (bool): Force rebuild of index

        Returns:
            bool: Success status
        """
        print("Initializing Academic Papers RAG System...")

        # Step 1: Create/load index
        if not self.create_index(force_rebuild=force_rebuild):
            print("Failed to create index")
            return False

        # Step 2: Setup query engine
        if not self.setup_query_engine():
            print("Failed to setup query engine")
            return False

        print("✓ Academic Papers RAG System initialized successfully")
        return True


def main():
    """Example usage of the Academic Papers RAG system."""
    # Initialize the RAG system
    rag = AcademicPapersRAG(
        papers_folder="papers/agents",
        chunk_size=512,
        chunk_overlap=50,
        similarity_top_k=3,
    )

    # Initialize the system
    if not rag.initialize_system():
        print("Failed to initialize RAG system")
        return

    # List indexed papers
    papers = rag.list_indexed_papers()
    print(f"\nIndexed papers ({len(papers)}):")
    for paper in papers:
        print(f"- {paper['file_name']}")
        print(f"  Title: {paper['title']}")
        print(f"  Authors: {paper['authors']}")
        print(f"  Pages: {paper['page_count']}")
        print()

    # Example queries
    example_queries = [
        "What are the main types of AI agents discussed in these papers?",
        "How do LLM-based agents differ from traditional AI agents?",
        "What are the current challenges in developing autonomous agents?",
        "What evaluation methods are used for AI agents?",
    ]

    print("Running example queries...")
    for query in example_queries:
        print(f"\n🔍 Query: {query}")
        result = rag.search_papers(query)

        if result["success"]:
            print(f"📝 Response: {result['response'][:300]}...")
            print(f"📚 Sources: {result['num_sources']}")
        else:
            print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()
