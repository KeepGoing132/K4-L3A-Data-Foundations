#!/usr/bin/env python3
"""Benchmark evaluation script for K4-L3A (VinUni Tuition dataset).

Meets Deliverable #3: bench.py + ket_qua_benchmark.txt.
Runs 5 benchmark queries and retrieves top-3 chunks per query with scores and metadata.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.chunking import RecursiveChunker
from src.embeddings import MockEmbedder
from src.models import Document
from src.store import EmbeddingStore

# 5 Câu hỏi đánh giá chuẩn (Benchmark Queries) cho chủ đề Học phí VinUni
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm?",
        "gold_answer": "815.850.000 VND / năm học (407.925.000 VND / kỳ học).",
        "gold_doc": "vinuni-hoc-phi-cu-nhan",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Chính sách hỗ trợ 35% học phí từ Vingroup áp dụng cho những ai và duy trì bao lâu?",
        "gold_answer": "Áp dụng tự động cho tất cả sinh viên trúng tuyển (Việt Nam và quốc tế) và duy trì suốt toàn bộ thời gian học chính thức.",
        "gold_doc": "vinuni-ho-tro-hoc-phi-35",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Học phí học lại theo tín chỉ tại VinUni được tính bằng bao nhiêu phần trăm?",
        "gold_answer": "Tính bằng 50% mức học phí chuẩn theo tín chỉ tương ứng của môn học đó.",
        "gold_doc": "vinuni-hoc-phi-tin-chi-hoc-lai",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Rút hồ sơ trước khi học kỳ bắt đầu thì được hoàn lại bao nhiêu phần trăm học phí?",
        "gold_answer": "Được hoàn trả 90% số học phí thực nộp của học kỳ đó.",
        "gold_doc": "vinuni-quy-dinh-nop-va-hoan-hoc-phi",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Mức chiết khấu giảm học phí cho con em là bao nhiêu?",
        "gold_answer": "Được giảm thêm 5% học phí niêm yết (cộng dồn với 35% thành 40%). Cần filter audience=student để tránh nhầm với chính sách nhân sự chung.",
        "gold_doc": "vinuni-chinh-sach-giam-hoc-phi-cbnv",
        "filter": {"audience": "student"},  # Yêu cầu bắt buộc của K4_VARIANT.md
    },
]


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


def run_benchmark():
    corpus_dir = Path("data/university")
    md_files = sorted(corpus_dir.glob("*.md"))
    print(f"=== KHỞI TẠO BENCHMARK: {corpus_dir} ===")
    print(f"Tìm thấy {len(md_files)} tài liệu nguồn.")

    # Chiến lược chia nhỏ (mỗi người có thể tùy chỉnh tham số chunk_size)
    chunker = RecursiveChunker(chunk_size=350)
    all_chunks: list[Document] = []

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(text)
        doc_id = metadata.get("doc_id", path.stem)

        # Cắt nhỏ phần thân văn bản
        chunks = chunker.chunk(body)
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{path.stem}#{i}",
                content=chunk_text,
                metadata={
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "source_path": str(path),
                },
            )
            all_chunks.append(chunk_doc)

    print(f"Đã chia nhỏ thành tổng cộng: {len(all_chunks)} chunks.")

    # Khởi tạo Vector Store
    embedder = MockEmbedder()
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    store.add_documents(all_chunks)
    print(f"Đã nạp {store.get_collection_size()} chunks vào EmbeddingStore thành công.\n")

    print("=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ 5 BENCHMARK QUERIES (TOP-3 RETRIEVAL)")
    print("=" * 80)

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        gold_ans = item["gold_answer"]
        filter_meta = item["filter"]

        print(f"\n[Query #{qid}]: \"{query}\"")
        print(f"  * Đáp án chuẩn (Gold Answer): {gold_ans}")
        if filter_meta:
            print(f"  * Bộ lọc (Metadata Filter): {filter_meta}")
            results = store.search_with_filter(query, top_k=3, metadata_filter=filter_meta)
        else:
            results = store.search(query, top_k=3)

        print("  * Top-3 Chunks tìm được:")
        for rank, r in enumerate(results, start=1):
            doc_origin = r["metadata"].get("doc_id", "N/A")
            score = r["score"]
            chunk_id = r["id"]
            snippet = r["content"][:130].replace("\n", " ").strip()
            print(f"    {rank}. [{chunk_id}] (doc_id: {doc_origin}, score: {score:.4f})")
            print(f"       Trích đoạn: \"{snippet}...\"")

    print("\n" + "=" * 80)
    print("HOÀN THÀNH CHẠY BENCHMARK!")
    print("=" * 80)


if __name__ == "__main__":
    import io
    output_buffer = io.StringIO()
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, output_buffer)
    try:
        run_benchmark()
    finally:
        sys.stdout = original_stdout
        Path("ket_qua_benchmark.txt").write_text(output_buffer.getvalue(), encoding="utf-8")
        print("\n>> Đã lưu toàn bộ kết quả benchmark vào file: ket_qua_benchmark.txt (UTF-8)")

