from multimodal_rag.embedding import build_vectorstore

vs = build_vectorstore([])

for index in [32, 33, 34]:
    result = vs._collection.get(
        where={"chunk_index": index},
        include=["documents", "metadatas"],
    )

    print(f"\n--- CHUNK {index} ---")

    if result["documents"]:
        print(result["documents"][0])
    else:
        print("Chunk not found")
