# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Ngọc Bảo  
**Mã học viên:** 2A202602951  
**Nhóm:** K4-L3A (Chủ đề: Dịch vụ & Quy định Đại học - Phân hệ Học phí VinUni)  
**Ngày:** 19/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding chỉ về cùng một hướng trong không gian nhiều chiều (góc giữa chúng xấp xỉ 0 độ). Về mặt ngữ nghĩa, điều này thể hiện hai đoạn văn bản có sự tương đồng nội dung rất cao, bất kể độ dài ngắn của câu.

**Ví dụ có độ tương tự CAO:**
- Câu A: *"Học phí ngành Bác sĩ Y khoa VinUni là bao nhiêu một năm?"*
- Câu B: *"Sinh viên theo học Y khoa VinUniversity cần đóng bao nhiêu tiền học phí mỗi năm?"*
- Tại sao tương đồng: Cả hai câu đều hỏi về mức kinh phí đào tạo hằng năm của cùng một ngành học (Y khoa), dù dùng các từ vựng và cấu trúc ngữ pháp khác nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: *"Hạn chót nộp học phí kỳ Mùa Thu của sinh viên là ngày nào?"*
- Câu B: *"Hệ thống máy tính phòng lab thư viện mở cửa vào khung giờ nào?"*
- Tại sao khác: Hai câu thuộc hai phạm trù thông tin hoàn toàn tách biệt (nghĩa vụ tài chính học vụ vs thời gian vận hành cơ sở vật chất phòng lab).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid đo độ dài hình học tuyệt đối giữa hai điểm vector nên bị ảnh hưởng rất mạnh bởi độ dài văn bản (văn bản dài chứa nhiều từ sẽ làm độ dài vector bị phóng to). Trong khi đó, độ tương tự cosine đã chuẩn hóa độ lớn vector về 1 và chỉ đo góc tạo bởi hướng vector, giúp so sánh chính xác sự tương đồng ngữ nghĩa giữa một câu truy vấn ngắn và một đoạn văn bản dài.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> $\text{Số lượng chunk} = \lceil \frac{\text{độ\_dài\_tài\_liệu} - \text{độ\_chồng\_chéo}}{\text{kích\_thước\_chunk} - \text{độ\_chồng\_chéo}} \rceil = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$  
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi `overlap=100`, số lượng chunk là: $\lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25$ chunks (tăng thêm 2 chunks).  
> Ta muốn tăng độ chồng chéo để bảo toàn mạch ngữ cảnh tại các ranh giới cắt, ngăn chặn việc một thông tin hoặc câu văn quan trọng bị chia cắt làm đôi ở điểm giao giữa 2 chunks, từ đó tăng độ chính xác khi hệ thống RAG truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi sử dụng regex lookbehind `r"(?<=[.!?])(?:\s+|\n)"` để bóc tách văn bản dựa trên ranh giới câu mà vẫn giữ nguyên được các dấu chấm câu hoàn chỉnh. Xử lý triệt để các edge cases như văn bản rỗng, khoảng trắng thừa bằng `.strip()`, sau đó nhóm tối đa `max_sentences_per_chunk` câu vào mỗi khối chunk hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo nguyên tắc chia để trị với danh sách phân tách ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Trường hợp cơ sở (base case) là khi văn bản ngắn hơn `chunk_size` hoặc đã hết danh sách phân tách; các đoạn văn bản dài hơn sẽ được cắt nhỏ bằng dấu phân tách hiện tại, sau đó ghép tuần tự trong giới hạn kích thước và gọi đệ quy với phân tách cấp thấp hơn nếu có phần quá cỡ.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi xây dựng cơ chế lưu trữ vector in-memory trong danh sách `_store`, mỗi phần tử gồm `id`, `content`, `metadata` và vector `embedding` được sinh ra từ `_embedding_fn`. Khi tìm kiếm (`search`), vector của chuỗi câu hỏi được tính tích vô hướng (dot product) với toàn bộ vector lưu trữ, sau đó sắp xếp giảm dần theo điểm tương đồng để trích xuất `top_k` kết quả có điểm cao nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi lựa chọn phương pháp tiền lọc (pre-filtering): quét và lọc danh sách các bản ghi thỏa mãn đồng thời tất cả các cặp khóa - giá trị trong `metadata_filter` trước, sau đó mới tính điểm tương đồng vector trên tập ứng viên đã lọc để tối ưu thời gian. Hàm `delete_document` xóa tất cả bản ghi có `id == doc_id` hoặc `metadata['doc_id'] == doc_id` và trả về `True` nếu số lượng phần tử giảm xuống.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tác tử gọi `store.search(question, top_k=top_k)` để trích xuất các đoạn văn bản có độ liên quan ngữ nghĩa cao nhất. Sau đó, ngữ cảnh được đưa vào mẫu prompt chuẩn: `f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context:"` và chuyển cho hàm `llm_fn` tạo ra câu trả lời dựa trên sự thật (grounded).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\D\learn_AI\K4-L3A-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Mức thu học phí Y khoa là bao nhiêu? | Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm? | Cao | 0.88 | Đúng |
| 2 | Quy định nộp học phí theo tín chỉ | Thủ tục mượn sách thư viện trường | Thấp | 0.05 | Đúng |
| 3 | Chính sách hỗ trợ học phí 35% từ Vingroup | Khoản tài trợ 35% tiền học cho tân sinh viên | Cao | 0.82 | Đúng |
| 4 | Rút hồ sơ trước học kỳ được trả bao nhiêu tiền? | Thời hạn đóng tiền và hoàn trả học phí | Cao | 0.74 | Đúng |
| 5 | Học phí học lại tín chỉ bị cấm thi | Giờ làm việc của phòng y tế học đường | Thấp | 0.03 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả ở Cặp 1 và Cặp 3 cho thấy mô hình embedding biểu diễn ngữ nghĩa dạng phân bố (distributed semantic representation) rất mạnh mẽ. Dù các từ đồng nghĩa như "mức thu học phí" và "tiền học", hay "hỗ trợ" và "tài trợ" không trùng khớp mặt chữ, vector embedding vẫn ánh xạ chúng về các tọa độ rất gần nhau trong không gian đặc trưng.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân trong gói `src` (sử dụng script `bench.py`):

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|---|---|---|---|---|
| 1 | Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm? | `vinuni-hoc-phi-cu-nhan`: Bác sĩ Y khoa học 6 năm, học phí 815.850.000 VND/năm (407.925.000 VND/kỳ). | 0.3439 | Có | Học phí Bác sĩ Y khoa là 815.850.000 VND/năm. |
| 2 | Chính sách hỗ trợ 35% học phí từ Vingroup áp dụng cho những ai và duy trì bao lâu? | `vinuni-ho-tro-hoc-phi-35`: Tự động áp dụng cho tất cả sinh viên trúng tuyển và duy trì suốt toàn bộ thời gian học chính thức. | 0.2087 | Có | Áp dụng cho cả sinh viên Việt Nam và quốc tế trong toàn khóa học. |
| 3 | Học phí học lại theo tín chỉ tại VinUni được tính bằng bao nhiêu phần trăm? | `vinuni-hoc-phi-tin-chi-hoc-lai`: Mức học phí học lại được tính bằng 50% mức học phí chuẩn theo tín chỉ. | 0.2874 | Có | Học phí học lại bằng 50% đơn giá tín chỉ chuẩn. |
| 4 | Rút hồ sơ trước khi học kỳ bắt đầu thì được hoàn lại bao nhiêu phần trăm học phí? | `vinuni-quy-dinh-nop-va-hoan-hoc-phi`: Rút hồ sơ trước ngày bắt đầu học kỳ được hoàn trả 90% số học phí thực nộp. | 0.1705 | Có | Được hoàn trả 90% số học phí thực nộp. |
| 5 | Mức chiết khấu giảm học phí cho con em là bao nhiêu? *(Lọc: audience=student)* | `vinuni-hoc-phi-cu-nhan`: Lọc chuẩn xác khối thông tin sinh viên thường, không bị lẫn sang văn bản của cán bộ nhân viên. | 0.2231 | Có | Sinh viên hưởng hỗ trợ 35% từ nhà sáng lập. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5**

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi học được kinh nghiệm thiết kế trường dữ liệu siêu dữ liệu (metadata schema) chuẩn xác, đặc biệt là việc tách trường `audience` thành `student` và `staff`. Kỹ thuật tiền lọc (metadata pre-filtering) giúp hệ thống RAG thu hẹp chính xác không gian tìm kiếm, ngăn chặn việc mô hình lấy nhầm các chính sách phúc lợi nhân sự để trả lời cho sinh viên thông thường.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
