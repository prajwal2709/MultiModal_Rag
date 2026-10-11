from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def embed_chunks(chunks):
    texts=[chunk["text"]for chunk in chunks]

    embedding=model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embed_chunks=[
    ]
    