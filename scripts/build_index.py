"""End-to-end indexing pipeline: PDFs on disk -> a searchable FAISS index.

Run as a script (see the CLI at the bottom), or import `build_index` to
call it from other code -- e.g. a future scheduled ingestion job.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import load_settings

from ingestion.chunker import Chunker
from ingestion.embeddings import Embedder
from ingestion.pdf_loader import PDFLoader, PDFParsingError
from vector_db.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


def build_index(
    papers_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    chunker: Chunker,
    dimension: int,
    save_every: int = 10,
) -> FaissVectorStore:
    """Parse, chunk, embed, and index every PDF in a directory.

    One bad file (corrupted, unreadable, or with no extractable text)
    is logged and skipped, rather than aborting the whole run -- a
    single broken download shouldn't block indexing everything else.

    Progress is checkpointed to `index_dir` every `save_every` files,
    and once more immediately if the run is interrupted (Ctrl+C) --
    embedding a large corpus on CPU can take hours, and losing all of
    it to one interruption is a real cost, not a minor inconvenience.

    Args:
        papers_dir: Directory containing downloaded PDF files.
        index_dir: Directory to write the resulting FAISS store to.
        embedder: Embedding model wrapper (see ingestion.embeddings).
        chunker: Chunking configuration (see ingestion.chunker).
        dimension: Embedding dimension; must match `embedder`'s model.
        save_every: Save a checkpoint after every N successfully
            indexed files, in addition to the final save. Set to 0 to
            disable checkpointing and only save once, at the end.

    Returns:
        The built FaissVectorStore, already saved to `index_dir`.

    Raises:
        KeyboardInterrupt: Re-raised after saving whatever was
            indexed so far, so the caller still sees the interruption
            (e.g. a CLI exits with the standard SIGINT status) while
            no completed work is lost.
    """
    loader = PDFLoader()
    store = FaissVectorStore(dimension=dimension)

    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", papers_dir)

    try:
        for processed_count, pdf_path in enumerate(pdf_paths, start=1):
            try:
                document = loader.load(pdf_path)
            except (FileNotFoundError, PDFParsingError) as exc:
                logger.warning("Skipping %s: %s", pdf_path.name, exc)
                continue

            chunks = chunker.split(document)
            if not chunks:
                logger.warning("Skipping %s: no extractable text", pdf_path.name)
                continue

            embedded_chunks = embedder.embed_chunks(chunks)
            store.add(chunks, embedded_chunks)
            logger.info("Indexed %s (%d chunks)", pdf_path.name, len(chunks))

            if save_every and processed_count % save_every == 0:
                store.save(index_dir)
                logger.info("Checkpoint: %d chunks saved to %s", store.size, index_dir)
    except KeyboardInterrupt:
        store.save(index_dir)
        logger.warning("Interrupted -- progress saved (%d chunks) to %s", store.size, index_dir)
        raise

    store.save(index_dir)
    logger.info("Saved index with %d chunks to %s", store.size, index_dir)
    return store


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from downloaded PDFs.")
    parser.add_argument("--papers-dir", type=Path, default=None, help="Overrides PAPERS_DIR")
    parser.add_argument("--index-dir", type=Path, default=None, help="Overrides INDEX_DIR")
    parser.add_argument("--chunk-size", type=int, default=None, help="Overrides CHUNK_SIZE")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overrides CHUNK_OVERLAP")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint every N files (default: 10; overrides SAVE_EVERY). Use 0 to disable.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()

    try:
        build_index(
            papers_dir=args.papers_dir or settings.papers_dir,
            index_dir=args.index_dir or settings.index_dir,
            embedder=Embedder(model_name=settings.embedding_model),
            chunker=Chunker(
                chunk_size=args.chunk_size or settings.chunk_size,
                overlap=args.chunk_overlap
                if args.chunk_overlap is not None
                else settings.chunk_overlap,
            ),
            dimension=settings.embedding_dimension,
            save_every=args.save_every if args.save_every is not None else settings.save_every,
        )
    except KeyboardInterrupt:
        # build_index() already saved progress before re-raising; exit
        # cleanly here instead of printing a scary traceback for what
        # was, from the user's side, just pressing Ctrl+C.
        print("\nInterrupted. Progress up to the last checkpoint was saved.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    _run_cli()
