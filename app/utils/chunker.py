"""
app/utils/chunker.py — Parse documents and split into overlapping text chunks.
Supports PDF, DOCX, Markdown, and plain text.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import tiktoken

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


# ── Document parsers ──────────────────────────────────────────────────────────

def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _parse_docx(data: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _parse_markdown(data: bytes) -> str:
    import markdown
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str):
            self._parts.append(data)

        def get_text(self) -> str:
            return " ".join(self._parts)

    html = markdown.markdown(data.decode("utf-8", errors="replace"))
    stripper = _Stripper()
    stripper.feed(html)
    return stripper.get_text()


SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "markdown"}

_PARSERS = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "doc": _parse_docx,
    "md": _parse_markdown,
    "markdown": _parse_markdown,
    "txt": lambda d: d.decode("utf-8", errors="replace"),
}


def parse_document(filename: str, data: bytes) -> str:
    """Return plain text from any supported document format."""
    ext = filename.rsplit(".", 1)[-1].lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        logger.warning("unsupported file type, falling back to plain text", ext=ext)
        return data.decode("utf-8", errors="replace")
    text = parser(data)
    logger.info("document parsed", ext=ext, char_count=len(text))
    return text


# ── Token-aware text splitter ─────────────────────────────────────────────────

class TokenAwareChunker:
    """
    Splits text into chunks of at most chunk_size tokens
    with chunk_overlap tokens between consecutive chunks.
    Splits on paragraph boundaries when possible.
    """

    ENCODING = "cl100k_base"  # matches text-embedding-3-large

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self._enc = tiktoken.get_encoding(self.ENCODING)

    def _tokenize(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def _decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def split(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[TextChunk] = []
        current_tokens: list[int] = []
        chunk_index = 0

        def _flush() -> TextChunk:
            nonlocal chunk_index
            decoded = self._decode(current_tokens)
            c = TextChunk(
                text=decoded,
                chunk_index=chunk_index,
                token_count=len(current_tokens),
                metadata=metadata or {},
            )
            chunk_index += 1
            return c

        for para in paragraphs:
            para_tokens = self._tokenize(para)

            if len(current_tokens) + len(para_tokens) > self.chunk_size:
                if current_tokens:
                    chunks.append(_flush())
                    current_tokens = current_tokens[-self.overlap:]

            current_tokens.extend(para_tokens)

        if current_tokens:
            chunks.append(_flush())

        logger.info(
            "document chunked",
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        return chunks