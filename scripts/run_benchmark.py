#!/usr/bin/env python3
"""Run benchmark, chunking comparison, and RAG evaluation on VinUni Tuition dataset."""

import glob
import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import KnowledgeBaseAgent
from src.chunking import ChunkingStrategyComparator, FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import MockEmbedder
from src.models import Document
from src.store import EmbeddingStore


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    yaml_block = parts[1]
    body = parts[2]
    metadata = {}
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata, body.strip()


def load_corpus(corpus_dir: str = "data/university") -> list[Document]:
    docs = []
    for p in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        path = Path(p)
        raw_text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw_text)
        docs.append(Document(id=meta.get("doc_id", path.stem), content=body, metadata=meta))
    return docs


def run_chunking_comparison(docs: list[Document]):
    print("=" * 70)
    print("PHẦN 1: SO SÁNH CÁC CHIẾN LƯỢC CHUNKING (Chunking Strategy Comparison)")
    print("=" * 70)
    comparator = ChunkingStrategyComparator()
    for doc in docs[:2]:  # Chạy trên 2 tài liệu mẫu
        print(f"\n📄 Tài liệu: {doc.id} ({doc.metadata.get('title', '')}) - Độ dài: {len(doc.content)} ký tự")
        results = comparator.compare(doc.content, chunk_size=300)
        print("-" * 65)
        print(f"{'Chiến lược (Strategy)':<32} | {'Số Chunk':<10} | {'Độ dài TB':<12}")
        print("-" * 65)
        for name, stats in results.items():
            print(f"{name:<32} | {stats['count']:<10} | {stats['avg_length']:<12.1f}")
        print("-" * 65)


def run_benchmark_queries(store: EmbeddingStore):
    print("\n" + "=" * 70)
    print("PHẦN 2: CHẠY 5 CÂU HỎI ĐÁNH GIÁ (5 Benchmark Queries & Retrieval)")
    print("=" * 70)

    queries = [
        {
            "id": 1,
            "query": "Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm?",
            "expected_doc": "vinuni-hoc-phi-cu-nhan",
            "filter": None,
        },
        {
            "id": 2,
            "query": "Chính sách hỗ trợ 35% học phí từ Vingroup áp dụng cho những ai và trong bao lâu?",
            "expected_doc": "vinuni-ho-tro-hoc-phi-35",
            "filter": None,
        },
        {
            "id": 3,
            "query": "Học phí học lại theo tín chỉ tại VinUni được tính bằng bao nhiêu phần trăm?",
            "expected_doc": "vinuni-hoc-phi-tin-chi-hoc-lai",
            "filter": None,
        },
        {
            "id": 4,
            "query": "Rút hồ sơ trước khi học kỳ bắt đầu thì được hoàn lại bao nhiêu học phí?",
            "expected_doc": "vinuni-quy-dinh-nop-va-hoan-hoc-phi",
            "filter": None,
        },
        {
            "id": 5,
            "query": "Mức chiết khấu giảm học phí cho con em là bao nhiêu?",
            "expected_doc": "vinuni-chinh-sach-giam-hoc-phi-cbnv",
            "filter": {"audience": "student"},  # CÂU BẮT BUỘC LỌC AUDIENCE: STUDENT
            "note": "Câu này có metadata_filter={'audience': 'student'} để chỉ lấy thông tin sinh viên thường!",
        },
    ]

    for q in queries:
        print(f"\n[Query {q['id']}]: \"{q['query']}\"")
        if q["filter"]:
            print(f"  🔍 Áp dụng Metadata Filter: {q['filter']} ({q.get('note', '')})")
            results = store.search_with_filter(q["query"], top_k=2, metadata_filter=q["filter"])
        else:
            results = store.search(q["query"], top_k=2)

        print(f"  🎯 Top-1 Retrieved: doc_id={results[0]['id']} (Score: {results[0]['score']:.4f})")
        print(f"  📝 Trích đoạn nội dung tìm được:")
        snippet = results[0]["content"][:180].replace("\n", " ")
        print(f"     \"{snippet}...\"")


def main():
    docs = load_corpus()
    print(f"Đã nạp thành công {len(docs)} tài liệu học phí VinUni vào hệ thống.")

    # 1. So sánh chunking
    run_chunking_comparison(docs)

    # 2. Tạo Vector Store và nhúng dữ liệu
    embedder = MockEmbedder()
    store = EmbeddingStore(collection_name="vinuni_tuition", embedding_fn=embedder)

    # Chunk từng tài liệu rồi nạp vào store
    chunker = RecursiveChunker(chunk_size=350)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, c in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc.id}_chunk_{i}",
                content=c,
                metadata=dict(doc.metadata),
            )
            all_chunks.append(chunk_doc)

    store.add_documents(all_chunks)
    print(f"\nVector Store đã chia nhỏ thành {store.get_collection_size()} chunks và lưu trữ thành công.")

    # 3. Chạy 5 câu hỏi đánh giá
    run_benchmark_queries(store)


if __name__ == "__main__":
    main()
