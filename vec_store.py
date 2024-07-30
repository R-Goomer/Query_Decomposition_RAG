from langchain_community.vectorstores import Chroma # type: ignore

def create_vs(chunks,embedding, name):
    return Chroma.from_texts(chunks, embedding, collection_name = name)