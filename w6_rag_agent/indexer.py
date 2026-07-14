import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'documents')

def index_documents():
  loader = DirectoryLoader(
      path=DOCUMENTS_DIR,
      glob='*.txt',
      loader_cls=TextLoader,
      loader_kwargs={'autodetect_encoding': True}
  )

  docs = loader.load()
  chroma_client = chromadb.PersistentClient(path="./chroma")
  collection = chroma_client.get_or_create_collection(name="w6_concepts")
  if collection.count() == 0:
    next_id = 0
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=600, chunk_overlap=100, length_function=len,
    )
    for doc in docs: 
      splitters = text_splitter.create_documents([doc.page_content])
      collection.add(
        documents=[chunk.page_content for chunk in splitters],
        metadatas=[{"source": doc.metadata['source']} for _ in splitters],
        ids=[str(next_id + i) for i in range(len(splitters))],
      )
      next_id += len(splitters)
      print(f"{doc.metadata['source']} indexed with {len(splitters)} chunks.")
      print(f"\nTotal: {next_id} chunks indexed across {len(docs)} documents.")

index_documents()