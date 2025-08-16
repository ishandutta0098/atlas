#!/usr/bin/env python3
"""
PDF Processor for Academic Papers

This module handles PDF processing for academic papers, extracting:
- Full text content
- Metadata (title, authors, abstract)
- Document structure (sections, citations)
- Reference information

Uses PyMuPDF for robust PDF text extraction and metadata parsing.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from llama_index.core import Document


class PaperPDFProcessor:
    """Processor for academic PDF papers with enhanced metadata extraction."""

    def __init__(self):
        """Initialize the PDF processor."""
        self.supported_extensions = {".pdf"}

    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF document.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict[str, str]: Extracted metadata
        """
        metadata = {
            "file_path": pdf_path,
            "file_name": Path(pdf_path).stem,
            "title": "",
            "authors": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "page_count": 0,
        }

        try:
            doc = fitz.open(pdf_path)
            pdf_metadata = doc.metadata

            # Extract basic metadata
            metadata.update(
                {
                    "title": pdf_metadata.get("title", "").strip(),
                    "authors": pdf_metadata.get("author", "").strip(),
                    "subject": pdf_metadata.get("subject", "").strip(),
                    "keywords": pdf_metadata.get("keywords", "").strip(),
                    "creator": pdf_metadata.get("creator", "").strip(),
                    "producer": pdf_metadata.get("producer", "").strip(),
                    "creation_date": pdf_metadata.get("creationDate", "").strip(),
                    "modification_date": pdf_metadata.get("modDate", "").strip(),
                    "page_count": doc.page_count,
                }
            )

            doc.close()

        except Exception as e:
            print(f"Warning: Could not extract metadata from {pdf_path}: {e}")

        return metadata

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text content from PDF with page-level information.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Tuple[str, List[Dict]]: Full text and page-level content
        """
        full_text = ""
        pages_content = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()

                # Clean up the text
                page_text = self._clean_text(page_text)

                if page_text.strip():  # Only add non-empty pages
                    pages_content.append(
                        {
                            "page_number": page_num + 1,
                            "text": page_text,
                            "char_count": len(page_text),
                        }
                    )

                    full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"

            doc.close()

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return "", []

        return full_text.strip(), pages_content

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text (str): Raw extracted text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page breaks and form feeds
        text = re.sub(r"[\f\r]", "\n", text)

        # Normalize line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remove trailing/leading whitespace
        text = text.strip()

        return text

    def extract_abstract(self, text: str) -> str:
        """
        Extract abstract from paper text using pattern matching.

        Args:
            text (str): Full paper text

        Returns:
            str: Extracted abstract
        """
        # Common abstract patterns
        abstract_patterns = [
            r"(?i)abstract\s*[:\-]?\s*\n\s*(.*?)(?=\n\s*(?:keywords|introduction|1\.|1\s+introduction))",
            r"(?i)abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords|introduction|1\.|1\s+introduction))",
            r"(?i)summary\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords|introduction|1\.|1\s+introduction))",
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:  # Minimum length check
                    return abstract

        return ""

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract major sections from the paper.

        Args:
            text (str): Full paper text

        Returns:
            List[Dict[str, str]]: List of sections with titles and content
        """
        sections = []

        # Common section patterns
        section_patterns = [
            r"(?i)(\d+\.?\s*)(introduction|related work|methodology|method|approach|implementation|results|evaluation|discussion|conclusion|future work)",
            r"(?i)(introduction|related work|methodology|method|approach|implementation|results|evaluation|discussion|conclusion|future work)\s*\n",
        ]

        # Find section boundaries
        section_matches = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            section_matches.extend(matches)

        # Sort by position and extract content
        section_matches.sort(key=lambda x: x.start())

        for i, match in enumerate(section_matches):
            section_title = match.group().strip()
            start_pos = match.end()

            # Find end position (next section or end of text)
            if i + 1 < len(section_matches):
                end_pos = section_matches[i + 1].start()
            else:
                end_pos = len(text)

            section_content = text[start_pos:end_pos].strip()

            if section_content and len(section_content) > 50:
                sections.append(
                    {
                        "title": section_title,
                        "content": section_content[:2000],  # Limit section length
                    }
                )

        return sections

    def process_pdf_to_document(self, pdf_path: str) -> Optional[Document]:
        """
        Process a PDF file and convert it to a LlamaIndex Document.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Optional[Document]: LlamaIndex Document object
        """
        try:
            # Extract metadata
            metadata = self.extract_pdf_metadata(pdf_path)

            # Extract text content
            full_text, pages_content = self.extract_text_from_pdf(pdf_path)

            if not full_text.strip():
                print(f"Warning: No text extracted from {pdf_path}")
                return None

            # Extract abstract and sections
            abstract = self.extract_abstract(full_text)
            sections = self.extract_sections(full_text)

            # Enhanced metadata (reduced for chunking)
            enhanced_metadata = {
                "file_name": metadata["file_name"],
                "title": (
                    metadata["title"][:100]
                    if metadata["title"]
                    else metadata["file_name"]
                ),
                "authors": (
                    metadata["authors"][:100] if metadata["authors"] else "Unknown"
                ),
                "page_count": metadata["page_count"],
                "has_abstract": bool(abstract),
                "document_type": "academic_paper",
            }

            # Create Document with full text and metadata
            document = Document(text=full_text, metadata=enhanced_metadata)

            return document

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return None

    def process_papers_folder(self, folder_path: str) -> List[Document]:
        """
        Process all PDF files in a folder.

        Args:
            folder_path (str): Path to folder containing PDF files

        Returns:
            List[Document]: List of processed documents
        """
        documents = []
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"Error: Folder {folder_path} does not exist")
            return documents

        pdf_files = list(folder_path.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return documents

        print(f"Processing {len(pdf_files)} PDF files from {folder_path}")

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            document = self.process_pdf_to_document(str(pdf_file))

            if document:
                documents.append(document)
                print(f"✓ Successfully processed: {pdf_file.name}")
            else:
                print(f"✗ Failed to process: {pdf_file.name}")

        print(
            f"Successfully processed {len(documents)} out of {len(pdf_files)} PDF files"
        )
        return documents


def main():
    """Example usage of the PDF processor."""
    processor = PaperPDFProcessor()

    # Process papers from the agents folder
    papers_folder = "papers/agents"
    documents = processor.process_papers_folder(papers_folder)

    print(f"\nProcessed {len(documents)} documents:")
    for doc in documents:
        metadata = doc.metadata
        print(f"- {metadata.get('file_name', 'Unknown')}")
        print(f"  Title: {metadata.get('title', 'No title')}")
        print(f"  Pages: {metadata.get('page_count', 0)}")
        print(f"  Abstract: {'Yes' if metadata.get('has_abstract') else 'No'}")
        print(f"  Text length: {metadata.get('text_length', 0):,} characters")
        print()


if __name__ == "__main__":
    main()
