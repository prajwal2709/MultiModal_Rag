
from multimodal_rag.pdf_proccessor import extract_pdf, PDF_DIR
from multimodal_rag.chunking import chunk_text
from multimodal_rag.embedding import build_vectorstore


def main():
    pdf_path = PDF_DIR / "attention-is-all-you-need.pdf"

    # Extract and chunk the PDF.
    text_pages, _ = extract_pdf(pdf_path)
    document_id = text_pages[0]["document_id"] if text_pages else "unknown"
    chunks = chunk_text(text_pages, document_id=document_id)

    print(f"Processed chunks: {len(chunks)}")

    # Index in the schema-v2 collection. The legacy pdf_text collection and
    # all files under CHROMA_DIR are deliberately left untouched.
    vectorstore = build_vectorstore(chunks)

    print(f"Collection: {vectorstore._collection.name}")
    print("Total records in Chroma:", vectorstore._collection.count())

    # Verify metadata.
    sample = vectorstore._collection.get(
        limit=5,
        include=["metadatas"],
    )

    print("\n--- Metadata check ---")
    for metadata in sample["metadatas"]:
        print(metadata)

    required = {"document_id", "page", "chunk_index", "source_type"}
    invalid = [
        item for item in sample["metadatas"]
        if not required.issubset(item)
    ]
    if invalid:
        raise RuntimeError(f"Indexed metadata is incomplete: {invalid}")


if __name__ == "__main__":
    main()
