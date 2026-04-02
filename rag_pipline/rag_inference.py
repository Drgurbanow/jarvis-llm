"""
RAG (Retrieval-Augmented Generation) pipeline for JARVIS assistant.
Uses FAISS vector store and BAAI/bge-base-en-v1.5 embeddings.
"""

import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

Settings.llm = None #Turn off openAI llm, cause i oonly need answer

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"C:\Users\user\.cache\huggingface\hub\models--BAAI--bge-base-en-v1.5",
    query_instruction="Represent this sentence for searching relevant passages: ",
    device="cuda",
)

documents = SimpleDirectoryReader(
    input_files=[r"C:\Users\user\Desktop\Rag_test.txt"],
    file_extractor={".pdf": PyMuPDFReader()} # others auto detect
).load_data()

parser = SentenceSplitter(chunk_size=180, chunk_overlap=50)
d=768
M=32
faiss_index = faiss.IndexHNSWFlat(d, M)
faiss_index.hnsw.efConstruction = 200

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                        transformations=[parser], show_progress=True
                                        )

#index.storage_context.persist(persist_dir="./my_rag")

"""query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("Jarvis, run a diagnostic on my sleep patterns for the last month. And don't sugarcoat it.")
"""
retriever_engine = index.as_retriever(similarity_top_k=3)
response = retriever_engine.retrieve("Jarvis, can you tell who is Francesco Cirillo, and what he created?")

print("\nОтвет:")

x = tuple((node.score, node.text) for node  in response)
for i in x:
    print(i)


