import fitz  # PyMuPDF


DEFAULT_CHUNK_SIZE = 500  # words per chunk


class PDFParseError(Exception):
    # raised when PDF extraction fails
    pass


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    # splits text into fixed word-count chunks
    words = text.split()
    return [
        " ".join(words[i: i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def parse_pdf(file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    # extracts text from all pages and returns as a list of chunks
    if not file_path or not file_path.strip():
        raise ValueError("file_path must be a non-empty string.")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise PDFParseError(f"Failed to open PDF '{file_path}': {e}") from e

    full_text_parts = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_text = page.get_text().strip()
            if page_text:  # skip blank pages silently
                full_text_parts.append(page_text)
    except Exception as e:
        raise PDFParseError(f"Failed to extract text from '{file_path}': {e}") from e
    finally:
        doc.close()

    if not full_text_parts:
        return []

    combined = "\n".join(full_text_parts)
    return _chunk_text(combined, chunk_size)