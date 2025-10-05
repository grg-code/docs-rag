import json
import re
from pathlib import Path
import frontmatter

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ---- Config ----
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
OUT_PATH = DATA_DIR / "chunks.jsonl"

# Heading-aware first, then chunk by length
HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT)
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def read_md_clean(path: Path):
    post = frontmatter.load(path)
    return post.content, post.metadata.get("source", None)


def clean_text(s: str) -> str:
    # Collapse excessive whitespace but preserve code blocks fencing
    # (simple and safe for MVP)
    return re.sub(r"\s+", " ", s).strip()


def chunk_one_file(rel: str, source_url: str):
    """Yield chunk records for a single markdown file identified by rel path."""
    md_path = RAW_DIR / rel
    if not md_path.exists():
        print(f"⚠️ Missing file: {md_path}")
        return

    text, front_meta = read_md_clean(md_path)
    header_docs = header_splitter.split_text(text)

    chunk_idx = 0
    for hd in header_docs:
        # Build a section path like "H1 > H2 > H3"
        headers = [hd.metadata.get(k) for k in ("h1", "h2", "h3", "h4") if hd.metadata.get(k)]
        section = " > ".join(headers) if headers else None

        # Further split this section by length for retrieval efficiency
        for piece in char_splitter.split_text(hd.page_content):
            body = clean_text(piece)
            if not body:
                continue
            yield {
                "id": f"{rel}::{chunk_idx:04d}",  # deterministic ID
                "rel": rel,
                "source": source_url,
                "section": section,
                "chunk_index": chunk_idx,
                "text": body,
            }
            chunk_idx += 1


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("manifest.jsonl not found. Run fetch first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUT_PATH.with_suffix(".tmp.jsonl")

    total_chunks = 0
    with MANIFEST_PATH.open("r", encoding="utf-8") as mf, tmp_path.open("w", encoding="utf-8") as out:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rel = entry["rel"]
            src = entry.get("source_url")

            for rec in chunk_one_file(rel, src):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    tmp_path.rename(OUT_PATH)
    print(f"✅ Saved {total_chunks} chunks → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
