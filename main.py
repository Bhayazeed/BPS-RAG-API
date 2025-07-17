import uvicorn
import json
import asyncio
import shutil
import uuid
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from operator import itemgetter # <-- Impor yang dibutuhkan untuk debug

# Import dari file lokal
import config
from api_models import QueryRequest, QueryResponse, IngestResponse, DocumentListResponse, DebugQueryResponse
from rag_components import load_or_create_rag_chain, format_docs_with_metadata

# Langchain & Docling imports untuk ingesti
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


# State aplikasi akan disimpan di sini, bukan sebagai variabel global kosong
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Mengelola proses startup dan shutdown aplikasi.
    """
    # Kode yang dijalankan sebelum aplikasi mulai menerima request (startup)
    print("Memulai proses startup server...")
    config.PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- PERUBAHAN PENTING ---
    # Pastikan fungsi load_or_create_rag_chain Anda mengembalikan semua komponen ini
    rag_chain, db, compression_retriever, prompt, llm, output_parser = load_or_create_rag_chain()
    
    # Simpan semua komponen di state aplikasi untuk digunakan nanti
    app_state["rag_chain"] = rag_chain
    app_state["db"] = db
    app_state["compression_retriever"] = compression_retriever
    app_state["prompt"] = prompt
    app_state["llm"] = llm
    app_state["output_parser"] = output_parser
    
    print("Proses startup selesai. Server siap menerima request.")
    
    yield # Aplikasi berjalan di sini
    
    # Kode yang dijalankan setelah aplikasi berhenti (shutdown)
    print("Memulai proses shutdown...")
    app_state.clear()
    print("Proses shutdown selesai.")



# Inisialisasi FastAPI dengan lifespan manager
app = FastAPI(
    title="BPS Riau RAG API",
    description="API untuk menjawab pertanyaan dokumen publikasi statistik BPS Riau.",
    version="1.0.0",
    lifespan=lifespan 
)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "*" # Izinkan semua untuk kemudahan debugging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kunci untuk mencegah beberapa proses ingesti berjalan bersamaan
ingestion_lock = asyncio.Lock()


def process_pdf_to_chunks(pdf_path: config.Path) -> List[LangchainDocument]:
    """Fungsi inti yang memproses satu PDF menjadi list of chunks."""
    raw_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_ID)
    hf_tokenizer = HuggingFaceTokenizer(tokenizer=raw_tokenizer, max_tokens=config.MAX_TOKENS_FINAL_FILTER)
    hybrid_chunker = HybridChunker(tokenizer=hf_tokenizer, merge_peers=True)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=raw_tokenizer, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr=True 
    pipeline_options.do_table_structure=True
    pipeline_options.ocr_options.use_gpu=True
    pipeline_options.table_structure_options.do_cell_matching=False

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = doc_converter.convert(pdf_path)
    if conv_res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise ValueError(f"Gagal mengonversi dokumen: {pdf_path.name} dengan status {conv_res.status}")

    langchain_docs = []
    docling_doc = conv_res.document
    document_id = str(uuid.uuid4())
    for i, chunk in enumerate(hybrid_chunker.chunk(dl_doc=docling_doc)):
        page_num = 0
        if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if hasattr(item, 'prov') and item.prov: page_num = item.prov[0].page_no; break
        cleaned_text = re.sub(r'GLYPH<[^>]+>', '', chunk.text).strip()
        cleaned_text = re.sub(r'https:\/\/riau\.bps\.go\.id\S*', '', cleaned_text).strip()
        if not cleaned_text: continue
        sub_texts = text_splitter.split_text(cleaned_text)
        for j, sub_text in enumerate(sub_texts):
            metadata = {"chunk_id": f"{document_id}_{i}_{j}", "document_id": document_id, "source_filename": pdf_path.name, "page_number": page_num}
            langchain_docs.append(LangchainDocument(page_content=sub_text, metadata=metadata))
    return langchain_docs

def get_all_chunks_from_json() -> List[dict]:
    """Membaca semua data chunk dari file chunks.json."""
    if not config.CHUNKS_JSON_PATH.exists(): return []
    with open(config.CHUNKS_JSON_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

# Endpoints API
@app.get("/", include_in_schema=False)
async def root(): return RedirectResponse(url="/docs")

@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(file: UploadFile = File(...)):
    """Mengunggah dan memproses satu file PDF untuk ditambahkan ke knowledge base."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file dengan format .pdf yang diterima.")
    
    async with ingestion_lock:
        all_chunks_data = get_all_chunks_from_json()
        processed_files = {chunk['metadata']['source_filename'] for chunk in all_chunks_data}
        if file.filename in processed_files:
            raise HTTPException(status_code=409, detail=f"File '{file.filename}' sudah pernah diproses.")

        file_path = config.PDF_STORAGE_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()

        try:
            new_chunks = process_pdf_to_chunks(file_path)
            if not new_chunks:
                raise HTTPException(status_code=422, detail="Tidak ada konten yang dapat diekstrak dari PDF.")

            db = app_state["db"]
            db.add_documents(new_chunks)
            db.save_local(folder_path=str(config.VECTOR_STORE_PATH), index_name=config.INDEX_NAME)
            
            new_chunks_data = [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in new_chunks]
            all_chunks_data.extend(new_chunks_data)
            with open(config.CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_chunks_data, f, ensure_ascii=False, indent=4)
            
            return IngestResponse(filename=file.filename, message="File berhasil diproses.", chunks_created=len(new_chunks))
        except Exception as e:
            if file_path.exists(): file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses file: {e}")

@app.get("/documents", response_model=DocumentListResponse)
async def get_processed_documents():
    """Mengembalikan daftar nama file dari semua dokumen yang telah diproses."""
    all_chunks = get_all_chunks_from_json()
    processed_files = sorted(list({chunk['metadata']['source_filename'] for chunk in all_chunks}))
    return DocumentListResponse(processed_documents=processed_files)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Menerima pertanyaan, memprosesnya melalui RAG chain, dan mengembalikan jawaban."""
    rag_chain = app_state.get("rag_chain")
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG chain belum siap.")
    try:
        answer = rag_chain.invoke({"question": request.question})
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {e}")

# --- ENDPOINT BARU UNTUK DEBUGGING ---
@app.post("/debug/ask", response_model=DebugQueryResponse, tags=["Debugging"])
async def debug_ask_question(request: QueryRequest):
    """
    Endpoint ini mengembalikan jawaban DAN konteks yang diambil
    untuk membantu proses debugging.
    """
    # Ambil komponen yang sudah dimuat dari state aplikasi
    retriever = app_state.get("compression_retriever")
    prompt = app_state.get("prompt")
    llm = app_state.get("llm")
    output_parser = app_state.get("output_parser")

    # Pastikan semua komponen sudah siap
    if not all([retriever, prompt, llm, output_parser]):
        raise HTTPException(status_code=503, detail="Service Unavailable: Komponen RAG belum siap.")

    try:
        question = request.question
        
        # Langkah 1: Ambil konteks secara manual menggunakan retriever
        print(f"DEBUG: Mengambil konteks untuk pertanyaan: '{question}'")
        retrieved_docs = retriever.invoke(question)
        retrieved_context = format_docs_with_metadata(retrieved_docs)
        print(f"DEBUG: Konteks yang berhasil diambil:\n---\n{retrieved_context}\n---")

        # Langkah 2: Buat jawaban berdasarkan konteks yang baru diambil
        chain = prompt | llm | output_parser
        answer = chain.invoke({"context": retrieved_context, "question": question})

        # Langkah 3: Kembalikan semuanya untuk dianalisis
        return DebugQueryResponse(
            question=question,
            retrieved_context=retrieved_context,
            answer=answer
        )
    except Exception as e:
        print(f"ERROR saat menjalankan /debug/ask: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
