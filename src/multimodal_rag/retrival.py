
from pathlib import Path

from multimodal_rag.embedding import build_vectorstore
from multimodal_rag.image_embedding import (
    collection as image_collection,
    processor,
    model,
    device,
    get_features,
)

import torch


# --------------------------------
# 1. Search relevant text
# --------------------------------



def retrieve_text_raw(query, k=3):
    """Return the top-k similarity results without neighbor expansion."""
    vectorstore = build_vectorstore([])

    results = vectorstore.similarity_search(query, k=k)

    items = []
    seen = set()
    for result in results:
        metadata = result.metadata
        identity = (
            metadata.get("document_id", "unknown"),
            metadata.get("chunk_index"),
        )
        if metadata.get("chunk_index") is None or identity in seen:
            continue
        seen.add(identity)
        items.append({
            "text": result.page_content,
            "document_id": metadata.get("document_id", "unknown"),
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "page_chunk_index": metadata.get("page_chunk_index", 0),
            "source_type": metadata.get("source_type", "text"),
            "retrieved_by": [query],
        })
    return items


def retrieve_text(query, k=3, neighbor_window=1):
    """Return matching chunks plus adjacent chunks in document order."""
    neighbor_window = max(0, int(neighbor_window))
    vectorstore = build_vectorstore([])

    results = vectorstore.similarity_search(query, k=k)
    collection = vectorstore._collection
    retrieved_chunks = {}

    for result in results:
        metadata = result.metadata
        chunk_index = metadata.get("chunk_index")
        document_id = metadata.get("document_id", "unknown")

        if chunk_index is None:
            continue

        # Preserve the vector result first, then add bounded same-document
        # neighbors. The old query used chunk_index alone and crossed PDFs.
        anchor_key = (document_id, chunk_index)
        retrieved_chunks.setdefault(anchor_key, {
            "text": result.page_content,
            "document_id": document_id,
            "page": metadata["page"],
            "chunk_index": chunk_index,
            "page_chunk_index": metadata.get("page_chunk_index", 0),
            "source_type": "text",
            "retrieved_by": [query],
            "is_neighbor": False,
        })

        for index in range(
            max(0, chunk_index - neighbor_window),
            chunk_index + neighbor_window + 1,
        ):
            if index == chunk_index:
                continue
            adjacent = collection.get(
                where={
                    "$and": [
                        {"document_id": document_id},
                        {"chunk_index": index},
                    ]
                },
                include=["documents", "metadatas"],
            )

            for text, metadata in zip(
                adjacent["documents"],
                adjacent["metadatas"],
            ):
                adjacent_index = int(metadata["chunk_index"])
                identity = (document_id, adjacent_index)
                retrieved_chunks.setdefault(identity, {
                    "text": text,
                    "document_id": document_id,
                    "page": metadata["page"],
                    "chunk_index": adjacent_index,
                    "page_chunk_index": metadata.get("page_chunk_index", 0),
                    "source_type": "text",
                    "retrieved_by": [query],
                    "is_neighbor": True,
                })

    # Anchors retain their semantic ranking; neighbors follow each anchor.
    rank = {
        (item.metadata.get("document_id", "unknown"), item.metadata.get("chunk_index")): pos
        for pos, item in enumerate(results)
    }
    return sorted(
        retrieved_chunks.values(),
        key=lambda item: (
            rank.get((item["document_id"], item["chunk_index"]), 10**9),
            item["is_neighbor"],
            item["page_chunk_index"],
        ),
    )


# --------------------------------
# 3. Combine both retrieval results
# --------------------------------


def retrieve_multimodal(query, text_k=3, image_k=3, neighbor_window=1):
    text_results = retrieve_text(query, text_k, neighbor_window=neighbor_window)

    # Search more images than we need so page-aware ranking
    # has candidates to work with.
    image_results = retrieve_images(query, k=6)

    # Pages supported by the retrieved text chunks
    relevant_pages = {
        item["page"] for item in text_results
    }

    # Prioritize images on pages found by text retrieval.
    # Keep CLIP distance as the tie-breaker.
    # CLIP first limits the candidate set; then a small page-distance penalty
    # can promote a visually similar figure adjacent to its relevant passage.
    # Page location never makes an image a candidate by itself.
    for item in image_results:
        item["text_page_distance"] = (
            min(abs(item["page"] - page) for page in relevant_pages)
            if relevant_pages else None
        )
        item["rank_score"] = item["distance"] + (
            0.035 * min(item["text_page_distance"], 2)
            if item["text_page_distance"] is not None else 0.0
        )
    image_results.sort(key=lambda item: (item["rank_score"], item["distance"]))


    # Assign stable IDs for this retrieval result.
    # T = text source, I = image source.
    for index, item in enumerate(text_results, start=1):
        item["source_id"] = f"T{index}"

    for index, item in enumerate(image_results[:image_k], start=1):
        item["source_id"] = f"I{index}"

    return {
        "query": query,
        "text": text_results,
        "images": image_results[:image_k],
    }

def retrieve_images(query, k=3):
    image_count = image_collection.count()
    if k <= 0 or image_count == 0:
        return []

    # Convert the query into a CLIP text embedding.
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

        # Handle CLIP model outputs that wrap the feature tensor.
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        text_features = text_features / text_features.norm(
            p=2, dim=-1, keepdim=True
        )

    query_embedding = text_features[0].cpu().numpy().tolist()

    # Search the existing image collection.
    results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, image_count),
        include=["metadatas", "distances"],
    )

    images = []

    if not results["ids"] or not results["ids"][0]:
        return images

    for image_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        images.append({
            "image_id": image_id,
            "document_id": metadata.get("document_id", "unknown"),
            "image_path": metadata["image_path"],
            "page": metadata["page"],
            "distance": distance,
            "source_type": "image",
        })

    images = [item for item in images if Path(item["image_path"]).is_file()]

    return images


# --------------------------------
# 4. Test the combined retrieval
# --------------------------------

if __name__ == "__main__":
    results = retrieve_text(
        "How many sub-layers does each Transformer encoder layer have?",
        k=3,
    )

    for chunk in results:
        print(
            f"\n--- Chunk {chunk['chunk_index']} "
            f"| Page {chunk['page']} ---"
        )
        print(chunk["text"])
