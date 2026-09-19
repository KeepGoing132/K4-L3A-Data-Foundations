#!/usr/bin/env python3
"""Interactive Web UI for VinUni Tuition RAG System using Streamlit."""

import glob
import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import MockEmbedder
from src.models import Document
from src.store import EmbeddingStore

# Page Configuration
st.set_page_config(
    page_title="VinUni Tuition RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Clean Modern Look)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .score-badge {
        background: #2563EB;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip().strip('"').strip("'")
    return metadata, body.strip()


@st.cache_data
def load_all_docs(folder: str = "data/university") -> list[dict]:
    results = []
    for file_path in sorted(glob.glob(os.path.join(folder, "*.md"))):
        p = Path(file_path)
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        results.append({
            "doc_id": meta.get("doc_id", p.stem),
            "title": meta.get("title", p.stem),
            "audience": meta.get("audience", "all"),
            "department": meta.get("department", "unknown"),
            "source_url": meta.get("source_url", ""),
            "content": body,
            "path": str(p),
        })
    return results


# Sidebar Controls
st.sidebar.header("⚙️ Cấu Hình RAG Pipeline")

raw_docs = load_all_docs()
st.sidebar.write(f"📚 **Kho tài liệu:** {len(raw_docs)} văn bản học phí")

# Chunking Strategy Selector
strategy_name = st.sidebar.selectbox(
    "Chiến lược chia nhỏ (Chunking):",
    options=["RecursiveChunker (Khuyên dùng)", "SentenceChunker", "FixedSizeChunker"],
)

chunk_size = st.sidebar.slider("Kích thước Chunk (ký tự):", min_value=150, max_value=800, value=350, step=50)

# Metadata Filter Selector
st.sidebar.subheader("🔍 Lọc Siêu Dữ Liệu (Metadata Filter)")
audience_filter = st.sidebar.selectbox(
    "Đối tượng (Audience):",
    options=["Tất cả (Không lọc)", "student (Sinh viên)", "staff (Cán bộ nhân viên)"],
)

top_k = st.sidebar.slider("Số lượng kết quả trích xuất (Top-K):", min_value=1, max_value=5, value=3)

# Build Store based on settings
if "Recursive" in strategy_name:
    chunker = RecursiveChunker(chunk_size=chunk_size)
elif "Sentence" in strategy_name:
    chunker = SentenceChunker(max_sentences_per_chunk=3)
else:
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=30)

store = EmbeddingStore(collection_name="ui_store", embedding_fn=MockEmbedder())
all_chunks = []
for d in raw_docs:
    chunks = chunker.chunk(d["content"])
    for i, c in enumerate(chunks):
        chunk_doc = Document(
            id=f"{d['doc_id']}#{i}",
            content=c,
            metadata={
                "doc_id": d["doc_id"],
                "title": d["title"],
                "audience": d["audience"],
                "department": d["department"],
                "source_url": d["source_url"],
            },
        )
        all_chunks.append(chunk_doc)

store.add_documents(all_chunks)

# Main UI Header
st.markdown('<div class="main-title">🎓 Hệ Thống Tra Cứu Học Phí Đại Học VinUni</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng RAG (Retrieval-Augmented Generation) thông minh dựa trên Vector Database & Semantic Search</div>', unsafe_allow_html=True)

# Status Bar
col_a, col_b, col_c = st.columns(3)
col_a.metric("Số tài liệu nguồn", f"{len(raw_docs)} files")
col_b.metric("Tổng số chunks đã index", f"{len(all_chunks)} chunks")
col_c.metric("Backend Embedding", "MockEmbedder (MD5/Norm)")

st.divider()

# Sample Query Buttons
st.write("💡 **Gợi ý câu hỏi nhanh:**")
col1, col2, col3 = st.columns(3)
quick_query = None
if col1.button("📌 Học phí Bác sĩ Y khoa?"):
    quick_query = "Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm?"
if col2.button("🎁 Chính sách hỗ trợ 35%?"):
    quick_query = "Chính sách hỗ trợ 35% học phí từ Vingroup áp dụng cho những ai?"
if col3.button("🔁 Học phí học lại tín chỉ?"):
    quick_query = "Học phí học lại theo tín chỉ tại VinUni được tính bằng bao nhiêu phần trăm?"

# User Query Input
user_query = st.text_input(
    "Nhập câu hỏi tra cứu học phí của bạn:",
    value=quick_query or "Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm?",
    placeholder="Ví dụ: Rút hồ sơ trước học kỳ được hoàn bao nhiêu tiền?",
)

btn_search = st.button("🚀 Tra cứu ngay", type="primary")

if btn_search or quick_query:
    metadata_filter = None
    if audience_filter == "student (Sinh viên)":
        metadata_filter = {"audience": "student"}
    elif audience_filter == "staff (Cán bộ nhân viên)":
        metadata_filter = {"audience": "staff"}

    with st.spinner("Đang tìm kiếm trong cơ sở tri thức..."):
        if metadata_filter:
            results = store.search_with_filter(user_query, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = store.search(user_query, top_k=top_k)

        # KnowledgeBaseAgent synthesis
        def mock_llm_answer(prompt: str) -> str:
            if not results:
                return "Không tìm thấy thông tin phù hợp trong tập tài liệu."
            top_content = results[0]["content"]
            return f"Dựa trên tài liệu chính thức của VinUni ({results[0]['metadata'].get('title')}):\n\n> {top_content[:300]}..."

        agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm_answer)
        answer = agent.answer(user_query, top_k=top_k)

    # Display Answer
    st.subheader("🤖 Câu Trả Lời Của Trợ Lý RAG:")
    st.success(answer)

    # Display Retrieved Chunks
    st.subheader(f"📑 Top-{len(results)} Đoạn Trích Dẫn Liên Quan Nhất (Retrieved Chunks):")
    for idx, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        score = r.get("score", 0.0)
        with st.expander(f"Top {idx} — {meta.get('title', 'Tài liệu')} (Score: {score:.4f})", expanded=(idx == 1)):
            st.markdown(f"**ID:** `{r.get('id')}` | **Đối tượng:** `{meta.get('audience')}` | **Phòng ban:** `{meta.get('department')}`")
            st.info(r.get("content", ""))
            if meta.get("source_url"):
                st.markdown(f"🔗 [Xem nguồn chính thức]({meta.get('source_url')})")
