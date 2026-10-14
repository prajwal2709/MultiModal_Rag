from pathlib import Path
import hashlib
import pymupdf
from multimodal_rag.chunking import chunk_text
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
IMAGE_DIR = PROJECT_ROOT / "data" / "extracted_image"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path):
    text_pages = []
    images = []
    pdf_path = Path(pdf_path).resolve()
    document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    document_image_dir = IMAGE_DIR / document_id[:16]
    document_image_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            try:
                text = page.get_text().strip()
            except Exception as error:
                raise RuntimeError(
                    f"Text extraction failed on page {page_number} of {pdf_path}"
                ) from error
            if text:
                text_pages.append({
                    "text": text,
                    "page": page_number,
                    "document_id": document_id,
                })

            for image_index, img in enumerate(page.get_images(), start=1):
                try:
                    image_bytes = pdf.extract_image(img[0])
                    image_id = hashlib.sha256(
                        f"{document_id}:{page_number}:{image_index}:".encode()
                        + image_bytes["image"]
                    ).hexdigest()
                    image_path = document_image_dir / (
                        f"page_{page_number}_image_{image_index}_{image_id[:8]}."
                        f"{image_bytes['ext']}"
                    )
                    image_path.write_bytes(image_bytes["image"])
                    images.append({
                        "document_id": document_id,
                        "page": page_number,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "source_type": "image",
                    })
                except Exception as error:
                    raise RuntimeError(
                        f"Image extraction failed for image {image_index} "
                        f"on page {page_number} of {pdf_path}"
                    ) from error

    return text_pages, images


if __name__ == "__main__":
    pdf_path = PDF_DIR / "attention-is-all-you-need.pdf"

    text_pages, images = extract_pdf(pdf_path)

    print("Extracted text pages:", len(text_pages))
    print("Extracted images:", len(images))


    # Test chunking
    document_id = text_pages[0]["document_id"] if text_pages else "unknown"
    chunks = chunk_text(text_pages, document_id=document_id)


    print("Total text chunks:", len(chunks))


    # Find the actual PDF chunks containing "sub-layers"
    print("\n========== SEARCHING ACTUAL PDF CHUNKS ==========")

    for i, chunk in enumerate(chunks):
        if "sub-layers" in chunk["text"].lower():
            print(f"\n--- Chunk {i} | Page {chunk['page']} ---")
            print(chunk["text"])


    #for i, chunk in enumerate(chunks[:3], start=1):
     #   print(f"\n--- Chunk {i} | Page {chunk['page']} ---")
      #  print(chunk["text"])
