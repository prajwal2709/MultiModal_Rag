from pathlib import Path
import pymupdf
from multimodal_rag.chunking import chunk_text
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
IMAGE_DIR = PROJECT_ROOT / "data" / "extracted_image"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path):
    text_pages = []
    images = []

    pdf = pymupdf.open(pdf_path)

    for page_number, page in enumerate(pdf, start=1):

        # Extract text
        text = page.get_text().strip()

        if text:
            text_pages.append({
                "text": text,
                "page": page_number
            })

        # Extract images
        for image_index, img in enumerate(
            page.get_images(), start=1
        ):
            image_bytes = pdf.extract_image(img[0])
            image_path = (
                IMAGE_DIR /
                f"page_{page_number}_image_{image_index}.{image_bytes['ext']}"
            )

            image_path.write_bytes(image_bytes["image"])
            images.append({
                "image_path": str(image_path),
                "page": page_number
            })

    pdf.close()

    return text_pages, images


if __name__ == "__main__":
    pdf_path = PDF_DIR / "attention-is-all-you-need.pdf"

    text_pages, images = extract_pdf(pdf_path)

    print("Extracted text pages:", len(text_pages))
    print("Extracted images:", len(images))


    # Test chunking
    chunks = chunk_text(text_pages)

    print("Total text chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} | Page {chunk['page']} ---")
        print(chunk["text"])
