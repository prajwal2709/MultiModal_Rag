
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# Configure embeddings through OpenRouter
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


TEXT_COLLECTION_NAME = "pdf_text_v2"


def build_vectorstore(chunks):
    vectorstore = Chroma(
        # Keep the previous schema intact; v2 records include provenance fields.
        collection_name=TEXT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    if not chunks:
        return vectorstore

    unique = {}
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        document_id = str(chunk.get("document_id", "unknown"))
        page = int(chunk["page"])
        chunk_index = int(chunk["chunk_index"])
        dedup_key = (document_id, page, text)
        record_id = hashlib.sha256(
            f"{document_id}:{page}:{chunk_index}:{text}".encode("utf-8")
        ).hexdigest()
        unique.setdefault(
            dedup_key,
            (record_id, chunk, text, document_id, page, chunk_index),
        )

    if not unique:
        return vectorstore

    records = list(unique.values())
    texts = [record[2] for record in records]
    metadatas = [
        {
            "document_id": record[3],
            "page": record[4],
            "chunk_index": record[5],
            "page_chunk_index": int(record[1].get("page_chunk_index", 0)),
            "source_type": "text",
        }
        for record in records
    ]
    ids = [record[0] for record in records]

    # Generate vectors using OpenRouter's embedding model.
    vectors = embeddings.embed_documents(texts)

    vectorstore._collection.upsert(
        ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=vectors,
)

    return vectorstore
