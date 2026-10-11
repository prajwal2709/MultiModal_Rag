from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text_pages):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n","\n"," ",""]
    )

    chunks=[]

    for page in text_pages:
        page_chunks=splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text":chunk,
                "page":page["page"]
            })
            
    return chunks