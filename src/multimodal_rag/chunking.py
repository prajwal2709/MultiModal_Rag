
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text_pages, document_id="unknown"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for page in text_pages:
        page_chunks = splitter.split_text(page["text"])

        for page_chunk_index, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "document_id": document_id,
                "page": int(page["page"]),
                "chunk_index": len(chunks),
                "page_chunk_index": page_chunk_index,
                "source_type": "text",
            })

    return chunks


if __name__ == "__main__":
    test_pages = [
        {
            "page": 3,
            "text": (
                "The encoder is composed of a stack of N = 6 "
                "identical layers. Each layer has two sub-layers: "
                "a multi-head self-attention mechanism and a "
                "position-wise fully connected feed-forward network. "
                "We employ a residual connection around each of "
                "the two sub-layers, followed by layer normalization."
            ),
        }
    ]

    result = chunk_text(test_pages)

    print(f"\nTotal chunks created: {len(result)}")
