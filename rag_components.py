import os
import torch
from typing import Any, List, Sequence
from operator import itemgetter

from unsloth import FastLanguageModel
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.callbacks.manager import Callbacks
from FlagEmbedding import FlagReranker

import config

class BgeReranker(BaseDocumentCompressor):
    model_config = {"arbitrary_types_allowed": True}
    reranker: Any
    top_n: int

    def __init__(self, model_name:str = config.RERANKER_MODEL_ID, top_n: int = config.RERANKER_TOP_N, **kwargs: Any):
        reranker_instance = FlagReranker(model_name, use_fp16=True)
        super().__init__(reranker=reranker_instance, top_n=top_n, **kwargs)

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Callbacks = None) -> Sequence[Document]:
        if not documents: return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.compute_score(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:self.top_n]]

def format_docs_with_metadata(docs: List[Document]) -> str:
    formatted_strings = [f"Kutipan dari file '{doc.metadata.get('source_filename', 'N/A')}' halaman {doc.metadata.get('page_number', 'N/A')}:\n{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(formatted_strings)

def load_or_create_rag_chain():
    """
    Memuat RAG chain. Jika vector store FAISS tidak ada, ia akan membuat yang kosong.
    Mengembalikan RAG chain dan objek FAISS DB.
    """
    print("1. Memuat model embedding...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID, model_kwargs={'device': 'cuda'})

    vector_store_full_path = config.VECTOR_STORE_PATH / f"{config.INDEX_NAME}.faiss"
    
    print(f"DEBUG: Checking for FAISS index at absolute path: {vector_store_full_path.resolve()}")

    if vector_store_full_path.exists():
        print(f"2. Memuat FAISS index yang sudah ada dari: {config.VECTOR_STORE_PATH}")
        db = FAISS.load_local(folder_path=str(config.VECTOR_STORE_PATH), index_name=config.INDEX_NAME, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        print("2. Index FAISS tidak ditemukan. Membuat vector store kosong...")
        dummy_doc = [Document(page_content="start")]
        db = FAISS.from_documents(dummy_doc, embeddings)
        db.delete([db.index_to_docstore_id[0]])
        db.save_local(folder_path=str(config.VECTOR_STORE_PATH), index_name=config.INDEX_NAME)
        print(f"   Vector store kosong berhasil dibuat di: {config.VECTOR_STORE_PATH}")

    #Change to your token here
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set. Please provide your Hugging Face token.")

    print(f"3. Memuat model LLM ({config.LLM_REPO_ID})...")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=config.LLM_REPO_ID, max_seq_length=8192, dtype=None, load_in_4bit=True, token=hf_token)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, return_full_text=False)
    llm = HuggingFacePipeline(pipeline=pipe)

    print("4. Menyiapkan retriever dengan reranker...")
    reranker = BgeReranker()
    retriever = db.as_retriever(search_kwargs={'k': config.RETRIEVER_SEARCH_K})
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)

    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Anda adalah asisten AI statistik BPS yang sangat akurat, teliti, dan berbasis fakta. Jawab pertanyaan pengguna HANYA berdasarkan "KONTEKS" yang diberikan.

# ATURAN MUTLAK DALAM MENJAWAB:
1. **SELALU JAWAB DALAM BAHASA INDONESIA**
2.  **100% Berbasis Konteks:** Jawaban Anda HARUS dan HANYA berasal dari informasi yang ada di dalam "KONTEKS". Dilarang keras menambahkan, menyimpulkan, atau menghitung informasi yang tidak tertulis secara eksplisit di dalam konteks.
3.  **Kutipan Sumber Wajib:** Setiap fakta, angka, atau pernyataan dalam jawaban Anda HARUS diakhiri dengan kutipan sumber yang jelas. Format kutipan harus: `(Sumber: [nama_file_lengkap], halaman [nomor_halaman])`.
4.  **Informasi Kutipan yang Akurat:** Nama file dan nomor halaman HARUS diambil secara persis dari informasi `Kutipan dari file '[nama_file_lengkap]' halaman [nomor_halaman]'...` yang menyertai setiap potongan konteks. Pastikan ekstensi file (misalnya, .pdf) disertakan.
5.  **Verifikasi Tahun (SANGAT PENTING):** Jika pertanyaan pengguna menyebutkan tahun spesifik (misal, "2024"), periksa apakah "KONTEKS" berisi data untuk tahun tersebut. Jika konteks berisi data dari tahun yang berbeda (misal, 2023), Anda WAJIB menjawab: "Informasi untuk tahun yang diminta (2024) tidak ditemukan dalam konteks yang tersedia. Konteks yang relevan berisi data untuk tahun 2023." lalu berikan kutipan sumbernya.
6.  **Penanganan Tabel:** Ekstrak nilai yang diminta dari tabel secara akurat. Jika nilai spesifik yang ditanyakan tidak ada, jangan mengarang jawaban.
7.  **TIDAK ADA KALKULASI:** Dilarang keras melakukan perhitungan apa pun (penjumlahan, perkalian, rata-rata, dll.) dari data di dalam konteks.
8.  **Informasi Tidak Ditemukan:** Jika pertanyaan TIDAK DAPAT dijawab sama sekali berdasarkan "KONTEKS", jawab dengan tegas: "Berdasarkan konteks yang diberikan, informasi spesifik mengenai hal tersebut tidak ditemukan."
9.  **Hindari Halusinasi:** Pastikan setiap fakta atau angka yang Anda sebutkan memiliki dukungan langsung dari konteks.
10. **Tidak Berulang:** Pastikan jawaban jelas dan dapat dipahami oleh Manusia dengan BAIK.

<|eot_id|><|start_header_id|>user<|end_header_id|>
# --- CONTOH JAWABAN YANG BAIK ---

# PERTANYAAN:
Berapa jumlah penduduk miskin di Kota Pekanbaru pada tahun 2023 dan berapa IPM Provinsi Riau?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Berdasarkan konteks yang diberikan:
- Jumlah penduduk miskin di Kota Pekanbaru pada tahun 2023 adalah 100,9 ribu jiwa (Sumber: statistik-kesejahteraan-rakyat-riau-2024.pdf, halaman 58).
- Indeks Pembangunan Manusia (IPM) Provinsi Riau pada tahun 2023 adalah 73,52 (Sumber: provinsi-riau-dalam-angka-2024.pdf, halaman 112).
<|eot_id|><|start_header_id|>user<|end_header_id|>

# --- PERMINTAAN SEBENARNYA ---

# KONTEKS:
{context}

# PERTANYAAN:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    prompt = PromptTemplate.from_template(template)
    rag_chain = ({"context": itemgetter("question") | compression_retriever | format_docs_with_metadata, "question": itemgetter("question")} | prompt | llm | StrOutputParser())
    
    print("--- RAG Chain Siap Digunakan! ---")
    return rag_chain, db, compression_retriever, prompt, llm, StrOutputParser()

