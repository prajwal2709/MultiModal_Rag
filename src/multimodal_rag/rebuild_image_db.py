"""Build the provenance-aware image collection without deleting legacy data."""

from multimodal_rag.image_embedding import (
    collection,
    embed_images,
    remove_duplicate_images,
    store_images,
)
from multimodal_rag.pdf_proccessor import PDF_DIR, extract_pdf


def main():
    pdf_path = PDF_DIR / "attention-is-all-you-need.pdf"
    _, images = extract_pdf(pdf_path)
    images = remove_duplicate_images(images)
    embedded = embed_images(images)
    store_images(embedded)

    sample = collection.get(limit=10, include=["metadatas"])
    required = {"document_id", "page", "image_id", "image_path", "source_type"}
    invalid = [
        metadata for metadata in sample["metadatas"]
        if not required.issubset(metadata)
    ]
    if invalid:
        raise RuntimeError(f"Indexed image metadata is incomplete: {invalid}")
    print("Image records:", collection.count())
    print("Verified metadata records:", len(sample["metadatas"]))


if __name__ == "__main__":
    main()
