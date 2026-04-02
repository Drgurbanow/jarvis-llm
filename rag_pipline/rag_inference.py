"""
RAG (Retrieval-Augmented Generation) pipeline for JARVIS assistant.
Uses FAISS vector store and BAAI/bge-base-en-v1.5 embeddings.
"""

import faiss, torch, os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

BASE_DIR = os.path.dirname(__file__)
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models/bge-base-en-v1.5")
DOCUMENTS_PATH = [os.path.join(BASE_DIR, "data/knowledge_base.txt")]
PERSIST_DIR = os.path.join(BASE_DIR, "rag_storage")

Settings.llm = None #Turn off openAI llm, cause i only need answer
device = "cuda" if torch.cuda.is_available() else "cpu"

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_PATH,
    query_instruction="Represent this sentence for searching relevant passages: ",
    device=device,
)

if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader(
        input_files=DOCUMENTS_PATH,
        file_extractor={".pdf": PyMuPDFReader()} # others auto detect
    ).load_data()
    
    parser = SentenceSplitter(chunk_size=180, chunk_overlap=50)
    d=768
    M=32
    faiss_index = faiss.IndexHNSWFlat(d, M)
    faiss_index.hnsw.efConstruction = 200
    
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(documents, 
                                            storage_context=storage_context,
                                            transformations=[parser], 
                                            show_progress=True
                                            )
    
    index.storage_context.persist(persist_dir=PERSIST_DIR)

retriever_engine = index.as_retriever(similarity_top_k=5)
response = retriever_engine.retrieve("Jarvis, can you tell who is Francesco Cirillo, and what he created?")

sorted_nodes = sorted(response, key=lambda x: x.score)
selected_nodes = sorted_nodes[:2]
context = "\n\n".join([node.text for node in selected_nodes])

if __name__ == "__main__":
    print("\nAnswer:")
    for node in selected_nodes:
        print(node.score, node.text)
    print(context)


