"""Local browser UI for the multimodal PDF RAG application.

Run with ``uv run python web_ui.py`` from the MultiModal_Rag directory.
The server listens on localhost only and uses the existing RAG pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
PDF_DIR = ROOT / "data" / "pdfs"
IMAGE_DIR = ROOT / "data" / "extracted_image"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
INDEXING_DOCUMENTS: set[str] = set()


def _safe_error_message(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    error_text = str(error)
    if status_code == 402 or "Error code: 402" in error_text:
        return (
            "OpenRouter has insufficient available credits for this request. "
            "The app now uses shorter answer and citation-check limits; "
            "add credits or raise this key's limit if the error continues."
        )
    if status_code == 401 or "OPENROUTER_API_KEY" in error_text:
        return "OpenRouter rejected the key. Check OPENROUTER_API_KEY in .env and restart the UI."
    return f"Processing failed ({type(error).__name__}). Check the UI server terminal for details."


def _json(handler: BaseHTTPRequestHandler, status: int, value: dict) -> None:
    payload = json.dumps(value).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _document_names() -> dict[str, str]:
    """Map content-derived PDF IDs to filenames already present in the library."""
    names = {}
    if not PDF_DIR.is_dir():
        return names
    for pdf_path in PDF_DIR.glob("*.pdf"):
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        names[digest] = pdf_path.name
    return names


def _indexed_document_ids() -> set[str]:
    """Read indexed document IDs without initializing the image model."""
    import chromadb

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    available = {getattr(item, "name", item) for item in client.list_collections()}
    indexed = set()
    for collection_name in ("pdf_text_v2", "pdf_images_v2"):
        if collection_name in available:
            records = client.get_collection(collection_name).get(include=["metadatas"])
            indexed.update(
                metadata["document_id"]
                for metadata in records.get("metadatas", [])
                if metadata and metadata.get("document_id")
            )
    return indexed


def _remove_document_index(document_id: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    available = {getattr(item, "name", item) for item in client.list_collections()}
    for collection_name in ("pdf_text_v2", "pdf_images_v2"):
        if collection_name in available:
            client.get_collection(collection_name).delete(where={"document_id": document_id})
    image_dir = (IMAGE_DIR / document_id[:16]).resolve()
    try:
        image_dir.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        return
    if image_dir.is_dir():
        shutil.rmtree(image_dir)


def _attach_document_names(result: dict) -> None:
    names = _document_names()
    for field in ("retrieved_sources", "sources"):
        for source in result.get(field, []):
            source["document_name"] = names.get(source.get("document_id"), "Unknown document")


def _index_pdf(pdf_path: Path, progress_callback=None) -> tuple[int, int]:
    from multimodal_rag.chunking import chunk_text
    from multimodal_rag.embedding import build_vectorstore
    from multimodal_rag.pdf_proccessor import extract_pdf

    pages, images = extract_pdf(pdf_path)
    if progress_callback:
        progress_callback("text_indexing")
    doc_id = pages[0]["document_id"] if pages else hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    chunks = chunk_text(pages, document_id=doc_id)
    store = build_vectorstore(chunks)
    image_count = 0
    if images:
        if progress_callback:
            progress_callback("image_indexing")
        from multimodal_rag.image_embedding import (
            embed_images,
            remove_duplicate_images,
            store_images,
        )

        embedded = embed_images(remove_duplicate_images(images))
        store_images(embedded)
        image_count = len(embedded)
    return len(chunks), image_count


def _serve_static(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        handler.send_error(404)
        return
    payload = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type in {"application/javascript"}:
        content_type += "; charset=utf-8"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    server_version = "FolioLocal/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            _serve_static(self, WEB_DIR / "index.html")
            return
        if parsed.path == "/styles.css":
            _serve_static(self, WEB_DIR / "styles.css")
            return
        if parsed.path == "/app.js":
            _serve_static(self, WEB_DIR / "app.js")
            return
        if parsed.path == "/api/status":
            _json(self, 200, {
                "key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
                "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
            })
            return
        if parsed.path == "/api/documents":
            docs = []
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            indexed_ids = _indexed_document_ids()
            for path in sorted(PDF_DIR.glob("*.pdf")):
                size = path.stat().st_size
                document_id = hashlib.sha256(path.read_bytes()).hexdigest()
                page_count = 0
                figure_count = 0
                try:
                    import pymupdf

                    with pymupdf.open(path) as pdf:
                        page_count = len(pdf)
                        figure_count = sum(len(page.get_images(full=True)) for page in pdf)
                except Exception:
                    pass
                docs.append({
                    "name": path.name,
                    "document_id": document_id,
                    "indexed": document_id in indexed_ids,
                    "indexing": document_id in INDEXING_DOCUMENTS,
                    "pages": page_count,
                    "figures": figure_count,
                    "size": f"{size / (1024 * 1024):.1f} MB" if size >= 1024 * 1024 else f"{max(1, size // 1024)} KB",
                })
            _json(self, 200, {"documents": docs})
            return
        if parsed.path == "/api/pdf-page":
            query = parse_qs(parsed.query)
            name = query.get("name", [""])[0]
            page_number = query.get("page", ["1"])[0]
            try:
                page_number = int(page_number)
                scale = min(2.0, max(0.2, float(query.get("scale", ["1"])[0])))
            except ValueError:
                _json(self, 400, {"error": "Page number or zoom is invalid."})
                return
            pdf_path = (PDF_DIR / name).resolve()
            try:
                pdf_path.relative_to(PDF_DIR.resolve())
            except ValueError:
                _json(self, 400, {"error": "Invalid PDF selection."})
                return
            if not name or not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                _json(self, 404, {"error": "PDF not found in the library."})
                return
            try:
                import pymupdf

                with pymupdf.open(pdf_path) as pdf:
                    if page_number < 1 or page_number > len(pdf):
                        _json(self, 404, {"error": "Page is outside this document."})
                        return
                    pixmap = pdf[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    payload = pixmap.tobytes("png")
            except Exception as error:
                print(f"PDF page rendering failed ({type(error).__name__}).")
                _json(self, 500, {"error": "This PDF page could not be rendered."})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "private, max-age=60")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/pdf":
            name = parse_qs(parsed.query).get("name", [""])[0]
            pdf_path = (PDF_DIR / name).resolve()
            try:
                pdf_path.relative_to(PDF_DIR.resolve())
            except ValueError:
                _json(self, 400, {"error": "Invalid PDF selection."})
                return
            if not name or not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                _json(self, 404, {"error": "PDF not found in the library."})
                return
            payload = pdf_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", "inline; filename=\"document.pdf\"")
            self.send_header("Cache-Control", "private, max-age=60")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/figures":
            name = parse_qs(parsed.query).get("name", [""])[0]
            pdf_path = (PDF_DIR / name).resolve()
            try:
                pdf_path.relative_to(PDF_DIR.resolve())
            except ValueError:
                _json(self, 400, {"error": "Invalid PDF selection."})
                return
            if not name or not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                _json(self, 404, {"error": "PDF not found in the library."})
                return
            document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            image_dir = (IMAGE_DIR / document_id[:16]).resolve()
            try:
                image_dir.relative_to(IMAGE_DIR.resolve())
            except ValueError:
                _json(self, 400, {"error": "Invalid figure folder."})
                return
            figures = []
            if image_dir.is_dir():
                for image_path in sorted(image_dir.rglob("*")):
                    if not image_path.is_file():
                        continue
                    match = re.search(r"page_(\d+)_image_", image_path.name)
                    figures.append({
                        "path": str(image_path.resolve()),
                        "page": int(match.group(1)) if match else None,
                        "name": image_path.name,
                    })
            _json(self, 200, {"figures": figures})
            return
        if parsed.path.startswith("/api/jobs/") or parsed.path.startswith("/api/answers/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                _json(self, 404, {"error": "Background job not found."})
            else:
                _json(self, 200, job)
            return
        if parsed.path == "/api/image":
            raw_path = parse_qs(parsed.query).get("path", [""])[0]
            path = Path(raw_path).resolve()
            try:
                path.relative_to(IMAGE_DIR.resolve())
            except ValueError:
                self.send_error(403, "Image path is outside the extracted image folder")
                return
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path not in {"/api/ask", "/api/upload", "/api/remove"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
            _json(self, 413, {"error": "Request is empty or exceeds the 25 MB PDF limit."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _json(self, 400, {"error": "Request must contain valid JSON."})
            return

        if self.path == "/api/remove":
            self._remove_document(payload)
            return

        if self.path == "/api/ask":
            question = str(payload.get("question", "")).strip()
            retrieval_hint = str(payload.get("retrieval_hint", "")).strip()[:3000]
            if not question:
                _json(self, 400, {"error": "Enter a question first."})
                return
            if len(question) > 2000:
                _json(self, 400, {"error": "Questions must be 2,000 characters or fewer."})
                return
            if not os.getenv("OPENROUTER_API_KEY"):
                _json(self, 400, {"error": "Set OPENROUTER_API_KEY in your .env file, then restart the UI."})
                return
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "running", "phase": "retrieving"}

            def answer_worker() -> None:
                def update_phase(phase: str) -> None:
                    with JOBS_LOCK:
                        if job_id in JOBS:
                            JOBS[job_id]["phase"] = phase

                try:
                    # Import in the worker so model initialization doesn't hold
                    # the browser's initial POST request open.
                    from multimodal_rag.answer_generator import generate_answer

                    result = generate_answer(
                        question,
                        progress_callback=update_phase,
                        retrieval_hint=retrieval_hint,
                    )
                    _attach_document_names(result)
                    with JOBS_LOCK:
                        JOBS[job_id] = {
                            "status": "complete",
                            "phase": "complete",
                            "result": result,
                        }
                except Exception as error:
                    status_code = getattr(error, "status_code", None)
                    print(f"RAG request failed ({type(error).__name__}, HTTP {status_code}).")
                    with JOBS_LOCK:
                        JOBS[job_id] = {
                            "status": "error",
                            "error": _safe_error_message(error),
                        }

            threading.Thread(target=answer_worker, daemon=True).start()
            _json(self, 202, {"job_id": job_id})
            return

        name = Path(str(payload.get("name", "document.pdf"))).name
        name = re.sub(r"[^\w.() -]", "_", name).strip(" .") or "document.pdf"
        if not name.lower().endswith(".pdf"):
            _json(self, 400, {"error": "Only PDF files can be added."})
            return
        try:
            raw = base64.b64decode(payload.get("data", ""), validate=True)
        except (ValueError, TypeError):
            _json(self, 400, {"error": "The uploaded PDF data is invalid."})
            return
        if len(raw) > MAX_UPLOAD_BYTES:
            _json(self, 413, {"error": "This PDF is larger than the 25 MB upload limit."})
            return
        if not raw.startswith(b"%PDF-"):
            _json(self, 400, {"error": "The selected file does not appear to be a PDF."})
            return
        if not os.getenv("OPENROUTER_API_KEY"):
            _json(self, 400, {"error": "Set OPENROUTER_API_KEY in your .env file, then restart the UI."})
            return
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        full_digest = hashlib.sha256(raw).hexdigest()
        digest = full_digest[:12]
        destination = PDF_DIR / f"{digest}-{name}"
        destination.write_bytes(raw)
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "running"}
            INDEXING_DOCUMENTS.add(full_digest)

        def worker() -> None:
            def update_phase(phase: str) -> None:
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]["phase"] = phase

            try:
                update_phase("extracting")
                text_count, image_count = _index_pdf(destination, progress_callback=update_phase)
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "status": "complete",
                        "phase": "complete",
                        "text_chunks": text_count,
                        "images": image_count,
                    }
            except Exception as error:
                print(f"PDF indexing failed ({type(error).__name__}, HTTP {getattr(error, 'status_code', None)}).")
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "status": "error",
                        "error": _safe_error_message(error),
                    }
            finally:
                with JOBS_LOCK:
                    INDEXING_DOCUMENTS.discard(full_digest)

        threading.Thread(target=worker, daemon=True).start()
        _json(self, 202, {"job_id": job_id})

    def _remove_document(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            _json(self, 400, {"error": "Choose a PDF to remove."})
            return
        requested_name = str(payload.get("name", ""))
        name = Path(requested_name).name
        if not name or name != requested_name:
            _json(self, 400, {"error": "Invalid PDF selection."})
            return
        pdf_path = (PDF_DIR / name).resolve()
        try:
            pdf_path.relative_to(PDF_DIR.resolve())
        except ValueError:
            _json(self, 400, {"error": "Invalid PDF selection."})
            return
        if not name.lower().endswith(".pdf") or not pdf_path.is_file():
            _json(self, 404, {"error": "That PDF is no longer in the library."})
            return

        document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        with JOBS_LOCK:
            if document_id in INDEXING_DOCUMENTS:
                _json(self, 409, {"error": "Wait for this PDF to finish indexing before removing it."})
                return
            # Identical files share an ID. Keep their records while a copy remains.
            duplicate_remains = any(
                hashlib.sha256(path.read_bytes()).hexdigest() == document_id
                for path in PDF_DIR.glob("*.pdf")
                if path != pdf_path
            )
            try:
                if not duplicate_remains:
                    _remove_document_index(document_id)
                pdf_path.unlink()
            except Exception as error:
                print(f"PDF removal failed ({type(error).__name__}).")
                _json(self, 500, {"error": "Could not fully remove this PDF. Check the UI server terminal and try again."})
                return
        _json(self, 200, {"removed": name})

    def do_DELETE(self) -> None:
        if urlparse(self.path).path != "/api/documents":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 4096:
            _json(self, 400, {"error": "Choose a PDF to remove."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _json(self, 400, {"error": "Request must contain valid JSON."})
            return
        self._remove_document(payload)


def main() -> None:
    host = os.getenv("RAG_UI_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_UI_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Folio is running at http://{host}:{port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Folio…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
