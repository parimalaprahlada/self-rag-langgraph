import os
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
REPO_IDENTIFIER = "karpathy/nanoGPT"  
# REPO_IDENTIFIER = "parimalaprahlada/stocksrepo"
print(f"DEBUG clone_url = https://github.com/{REPO_IDENTIFIER}")
loader = GitLoader(
    clone_url=f"https://github.com/{REPO_IDENTIFIER}",
    repo_path="./code_data/my_repo",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size = 1500, chunk_overlap = 100
)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = loader.load()
print(f"Total docs loaded: {len(docs)}")
# for d in docs[:20]:
#     print(d.metadata.get("source"))

##chunk the code into chunks of 10000 characters maximum, and I remove any documents with more than 50000 characters.
docs = [doc for doc in docs if len(doc.page_content)<50000]
docs = python_splitter.split_documents(docs) 

print(f"Ingested {len(docs)} chunks")   # sanity check count

vectorstore = Chroma(
    collection_name="rag-chroma", 
    embedding_function=embedding_function,
    persist_directory="./chroma_langchain.db"
)

vectorstore.add_documents(documents = docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
