
import hashlib
from pathlib import Path
from IPython.display import display
import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from multimodal_rag.pdf_proccessor import (
    extract_pdf,
    PDF_DIR,
)

# -----------------------------
# 1. Configuration
# -----------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Separate collection for CLIP image embeddings
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    # Keep legacy path-only records separate from provenance-complete records.
    name="pdf_images_v2",
    metadata={"hnsw:space": "cosine"},
)


# -----------------------------
# 2. Handle CLIP model outputs
# -----------------------------

def get_features(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs

    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output

    raise TypeError(f"Unexpected CLIP output: {type(outputs)}")


# -----------------------------
# 3. Generate image embeddings
# -----------------------------

def embed_images(images, batch_size=8):
    embedded_images = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        valid_images = []
        valid_metadata = []

        for item in batch:
            try:
                with Image.open(item["image_path"]) as image:
                    valid_images.append(image.convert("RGB"))

                valid_metadata.append(item)

            except (OSError, ValueError) as error:
                print("Skipping image:", item["image_path"], error)

        if not valid_images:
            continue

        inputs = processor(
            images=valid_images,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            features = get_features(outputs)
            features = torch.nn.functional.normalize(
                features, p=2, dim=-1
            )

        vectors = features.cpu().tolist()

        for item, vector in zip(valid_metadata, vectors):
            embedded_images.append({
                **item,
                "embedding": vector,
            })

    return embedded_images


def remove_duplicate_images(images):
    unique_images = []
    seen_hashes = set()

    for item in images:
        try:
            with Image.open(item["image_path"]) as image:
                image = image.convert("RGB")
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        except (OSError, ValueError) as error:
            print(f"Skipping unavailable or invalid image {item['image_path']}: {error}")
            continue

        identity = (item.get("document_id", "unknown"), image_hash)
        if identity not in seen_hashes:
            seen_hashes.add(identity)
            unique_images.append(item)

    return unique_images
# -----------------------------
# 4. Save embeddings in ChromaDB
# -----------------------------

def store_images(embedded_images):
    if not embedded_images:
        print("No images to store.")
        return

    ids = [
        item.get("image_id") or hashlib.sha256(
            f"{item.get('document_id', 'unknown')}:{item['image_path']}".encode("utf-8")
        ).hexdigest()
        for item in embedded_images
    ]

    vectors = [
        item["embedding"]
        for item in embedded_images
    ]

    metadatas = [
        {
            "document_id": item.get("document_id", "unknown"),
            "image_id": item.get("image_id", ids[index]),
            "image_path": item["image_path"],
            "page": item["page"],
            "source_type": "image",
        }
        for index, item in enumerate(embedded_images)
    ]

    documents = [
        f"Image extracted from PDF page {item['page']}"
        for item in embedded_images
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )

    print("Images processed:", len(embedded_images))
    print("Total records in image collection:", collection.count())


# -----------------------------
# 5. Search images using text
# -----------------------------

def search_images(query, k=6):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = get_features(outputs)
        features = torch.nn.functional.normalize(
            features, p=2, dim=-1
        )

    query_vector = features.cpu().tolist()[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, collection.count()),
        include=["metadatas", "distances", "documents"],
    )

    for i, metadata in enumerate(results["metadatas"][0]):
        print(f"\nResult {i + 1}")
        print("Page:", metadata["page"])
        print("Image:", metadata["image_path"])
        print("Cosine distance:", results["distances"][0][i])

    # Open the retrieved image in macOS Preview
        import subprocess
        subprocess.run(["open", metadata["image_path"]], check=False)



# -----------------------------
# 6. Run the pipeline
# ---
if __name__ == "__main__":
    pdf_path = PDF_DIR / "attention-is-all-you-need.pdf"
    text_pages, images = extract_pdf(pdf_path)
    images = remove_duplicate_images(images)

    print("Unique images:", len(images))

    search_images(
    "The Transformer architecture diagram showing the encoder and decoder",
    k=6,
    )

    for item in images:
        print(
            f"Page {item['page']}: "
            f"{item['image_path']}"
        )


        import subprocess
        subprocess.run(
            ["open", item["image_path"]],
            check=False,
        )
