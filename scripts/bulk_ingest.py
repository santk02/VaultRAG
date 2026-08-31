#!/usr/bin/env python3
"""
Bulk document ingestion script for VaultRAG.

This script processes multiple documents from a directory and uploads them
to the VaultRAG system.
"""

import asyncio
import sys
from pathlib import Path
from typing import List
import aiofiles


async def ingest_directory(directory: str, file_extensions: List[str] = None):
    """
    Ingest all documents from a directory.

    Args:
        directory: Path to directory containing documents
        file_extensions: List of file extensions to process (default: .pdf, .docx)
    """
    # NOTE: upload logic below is a placeholder (prints intent, doesn't call the API) —
    # wire it up with an httpx.AsyncClient POST to /v1/documents/upload if bulk-loading
    # data/sample_docs against a running server is needed.
    if file_extensions is None:
        file_extensions = [".pdf", ".docx"]

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # Find all matching files
    files = []
    for ext in file_extensions:
        files.extend(dir_path.glob(f"*{ext}"))

    if not files:
        print(f"No files found with extensions {file_extensions} in {directory}")
        return

    print(f"Found {len(files)} files to process")

    # Process each file
    for file_path in files:
        print(f"Processing: {file_path.name}")

        try:
            # Read file content
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            # Here you would implement the actual upload logic
            # For now, this is a placeholder
            print(f"  - File size: {len(content)} bytes")
            print("  - Would upload to: /v1/documents/upload")

        except Exception as e:
            print(f"  - Error: {str(e)}")

    print("\nBulk ingestion complete!")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python bulk_ingest.py <directory> [extensions]")
        print("Example: python bulk_ingest.py ./data/sample_docs .pdf,.docx")
        sys.exit(1)

    directory = sys.argv[1]
    extensions = None

    if len(sys.argv) > 2:
        extensions = [ext.strip() for ext in sys.argv[2].split(",")]

    asyncio.run(ingest_directory(directory, extensions))


if __name__ == "__main__":
    main()
