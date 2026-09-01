"""Extract plain text from uploaded case files (PDF, DOCX, TXT/MD)."""
import io
from typing import Tuple

from pypdf import PdfReader
from docx import Document


def extract_text(filename: str, data: bytes) -> Tuple[str, str]:
    """Returns (text, note). note is a warning string if extraction was partial."""
    lower = filename.lower()
    try:
        if lower.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages).strip()
            if not text:
                return "", "PDF appears to be scanned/image-only; no extractable text found (OCR not run)."
            return text, ""
        if lower.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            paras = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    paras.append(" | ".join(c.text for c in row.cells))
            return "\n".join(paras).strip(), ""
        if lower.endswith((".txt", ".md", ".csv")):
            return data.decode("utf-8", errors="replace").strip(), ""
        return "", f"Unsupported file type for text extraction: {filename}. " \
                   "Supported: PDF, DOCX, TXT, MD, CSV."
    except Exception as e:  # noqa: BLE001
        return "", f"Failed to extract text from {filename}: {e}"
