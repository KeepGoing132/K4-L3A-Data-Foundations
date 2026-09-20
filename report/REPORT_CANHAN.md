# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Ngọc Bảo  
**Mã học viên:** 2A202602951  
**Nhóm:** K4-L3A (Chủ đề: Dịch vụ & Quy định Đại học — Học phí, Học bổng & Hỗ trợ tài chính VinUni)  
**Phân công trong nhóm:** Thành viên 3 (Phần 3: Nghiên cứu Baseline & Chiến lược Chunker theo Cấu trúc Heading / Structure-Aware)  
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding cùng chỉ về một hướng trong không gian nhiều chiều (góc $\theta$ giữa hai vector xấp xỉ 0 độ). Về mặt ngữ nghĩa, điều này thể hiện hai đoạn văn bản có sự tương đồng nội dung rất cao, bất kể độ dài ngắn hay số lượng từ vựng của chúng chênh lệch nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: *"Học phí niêm yết một năm của chương trình Cử nhân Điều dưỡng là bao nhiêu?"*
- Câu B: *"Sinh viên ngành Điều dưỡng tại VinUniversity phải nộp bao nhiêu tiền học phí mỗi năm?"*
- *Tại sao tương đồng:* Cả hai câu đều hỏi về mức kinh phí đào tạo hằng năm của cùng một ngành học (Điều dưỡng tại VinUni), dù sử dụng từ ngữ và cú pháp hoàn toàn khác nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: *"Thời hạn nộp hồ sơ xin hỗ trợ tài chính cho học kỳ mùa Thu là khi nào?"*
- Câu B: *"Quy định mượn sách và thời gian mở cửa thư viện trường đại học."*
- *Tại sao khác:* Hai câu thuộc hai phạm trù dịch vụ hoàn toàn khác biệt (hỗ trợ tài chính/học bổng vs cơ sở vật chất/thư viện), không có sự giao thoa ngữ nghĩa.

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

Giải thích cách tôi lập trình các thành phần cốt lõi trong gói `src/`:

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng biểu thức chính quy (regex) `(?<=[.!?])\s+` (cơ chế look-behind) để chia nhỏ văn bản dựa trên ranh giới kết thúc câu mà không làm mất dấu chấm câu. Sau đó, gom các câu lại vào từng chunk sao cho số lượng câu trong mỗi chunk không vượt quá `max_sentences_per_chunk`. Xử lý trường hợp chuỗi rỗng và loại bỏ các khoảng trắng thừa ở hai đầu câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo nguyên lý đệ quy chia nhỏ dần theo danh sách dấu phân cách có thứ tự ưu tiên: `["\n\n", "\n", ". ", " ", ""]`.
> - *Trường hợp cơ sở (Base case):* Nếu độ dài đoạn văn nhỏ hơn hoặc bằng `chunk_size` hoặc đã duyệt hết danh sách phân cách, dừng đệ quy. 
> - *Trường hợp đệ quy:* Tách đoạn văn theo dấu phân cách hiện tại; nếu đoạn nào vẫn dài hơn `chunk_size`, gọi đệ quy xuống dấu phân cách nhỏ hơn tiếp theo. Sau đó, gom các mảnh nhỏ liền kề lại sát giới hạn `chunk_size` để tạo thành chunk hoàn chỉnh.

**Chiến lược R3 — `StructureAwareChunker`**:
> Tôi phân tích cây heading Markdown (`#` đến `######`) và ưu tiên giữ nguyên một nhánh nội dung nếu còn trong ngân sách ký tự. Mỗi chunk được gắn header ngữ cảnh gồm tên tài liệu và breadcrumb của mục; bảng Markdown chỉ được tách theo hàng và lặp lại hàng tiêu đề. Nhờ vậy, số tiền hoặc điều kiện không bị tách khỏi tên mục đang áp dụng. Trên 10 tài liệu nhóm, chiến lược `structure_tree` tạo **70 chunks**, dài trung bình **989 ký tự** (min 189, max 2.140).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu văn bản, metadata và vector do `embedding_fn` sinh ra trong danh sách in-memory. Nếu embedder hỗ trợ `prefetch`, toàn bộ chunk được nhúng theo lô trước khi tạo record. Khi tìm kiếm, hệ thống nhúng truy vấn, tính tích vô hướng với từng vector tài liệu, sắp xếp giảm dần và trả về `top_k`; với các backend đang dùng, vector đã được chuẩn hóa L2 nên tích vô hướng tương đương cosine similarity.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> - `search_with_filter`: Thực hiện lọc siêu dữ liệu (metadata) **TRƯỚC** khi tính toán độ tương tự (Pre-filtering) để tối ưu hiệu năng tính toán và loại trừ dữ liệu không thuộc đối tượng quan tâm. Tài liệu chỉ được đưa vào so khớp cosine nếu tất cả các cặp key-value trong `metadata_filter` đều khớp chính xác với `document.metadata`.
> - `delete_document`: Lọc lại danh sách tài liệu trong store bằng list comprehension, chỉ giữ lại các tài liệu có `metadata.get('doc_id') != doc_id`. Trả về `True` nếu số lượng tài liệu giảm đi, ngược lại trả về `False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Truy xuất `top_k` đoạn liên quan từ `EmbeddingStore.search()`, nối nguyên văn các đoạn thành phần `Context`, thêm câu hỏi và chỉ dẫn “chỉ trả lời dựa trên ngữ cảnh”, rồi gọi `llm_fn`. Pipeline benchmark nâng cao trong `rag/pipeline.py` còn đánh số nguồn `[1]`, `[2]`, đưa `doc_id`, `audience`, `document_version` vào nhãn trích dẫn và trả về thông báo không tìm thấy khi retrieval rỗng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Đã hoàn thành toàn bộ các hàm TODO trong gói `src/` và vượt qua 100% các bài kiểm thử tự động của giảng viên.

### Kết Quả Kiểm Thử (Test Results)

```text
> python -m unittest discover -s tests -v
...
----------------------------------------------------------------------
Ran 42 tests

OK
```

**Số lượng bài test vượt qua (pass):** **42 / 42 tests (100%)**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Đo lường độ tương tự cosine giữa các câu thử nghiệm trên chủ đề quy định học phí VinUni:

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | "Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm?" | "Mức học phí hàng năm của ngành Y khoa tại trường VinUni." | Cao | **0,4117** | Đúng |
| 2 | "Rút hồ sơ trước học kỳ được hoàn bao nhiêu phần trăm?" | "Quy định về tỷ lệ hoàn trả tiền học khi thôi học sớm." | Cao | **0,4629** | Đúng |
| 3 | "Chính sách giảm giá học phí cho con cán bộ nhân viên." | "Điều kiện mượn giáo trình tại thư viện trường." | Thấp | **0,1885** | Đúng |
| 4 | "Thời hạn nộp học phí học kỳ mùa thu là ngày nào?" | "Hạn chót đóng tiền học kỳ 1 của sinh viên đại học." | Cao | **0,5675** | Đúng |
| 5 | "Học bổng 100% yêu cầu điểm GPA tối thiểu bao nhiêu?" | "Thực đơn món ăn tại căng tin ký túc xá sinh viên." | Thấp | **0,2058** | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Lần chạy hiện tại dùng `offline lexical hashing (512d)` để có thể tái lập mà không cần API key. Cặp 2 vẫn đạt 0,4629 dù cách diễn đạt khác nhau vì còn chia sẻ các tín hiệu như “học”, “hoàn/hoàn trả”; ngược lại hai cặp thấp vẫn có điểm khoảng 0,19–0,21 do từ chung và va chạm feature-hashing. Điều này cho thấy baseline từ vựng phân biệt được chủ đề ở mức cơ bản nhưng không thay thế được embedding ngữ nghĩa đa ngôn ngữ.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

### Chiến lược của tôi: Chunker theo Cấu trúc Heading / Structure-Aware (`structure_tree`)
- **Vai trò:** Thành viên 3 (Phần 3 trong nhóm — phụ trách Heading-based & Structure-Aware Chunking).
- **Ý tưởng thiết kế:** Thay vì cắt cứng theo ký tự hoặc số câu, tôi phân tích cây heading của văn bản quy định. Một mục được giữ nguyên nếu cả nhánh còn trong ngân sách; mục lớn mới được tách theo block, hàng bảng hoặc list item. Mỗi phần đều lặp header ngữ cảnh theo dạng `Tài liệu > Phần > Mục`.
- **Cấu hình thực nghiệm:** `StructureAwareChunker(mode="tree", max_chars=2200, min_chars=400)`, sinh 70 chunks từ 10 tài liệu. Benchmark cá nhân dùng backend ngoại tuyến tái lập (`offline lexical hashing 512d` + câu trả lời trích xuất), không bật BM25, HyDE hay reranker để đo riêng ảnh hưởng của chunking.

> `REPORT_NHOM.md` lưu số của lần chạy trước bằng `text-embedding-3-small` (fixed 7, sentence 6, structure 8). Báo cáo cá nhân này dùng lần chạy offline mới (fixed 6, sentence 5, structure 8), vì vậy điểm baseline khác nhẹ nhưng kết luận về R3 không thay đổi.

### Kết quả chạy 5 Benchmark Queries với Chiến lược của tôi (`structure_tree`):

| # | Câu hỏi (Query) | Top-1 Chunk tìm được | Score | Điểm (/2) | Trả lời đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| **Q1** | Học phí niêm yết Cử nhân Điều dưỡng là bao nhiêu? | `quy-dinh-tai-chinh-bieu-phi#0` [student] — mục *Học phí niêm yết* | 0,4029 | **2 / 2** | Có — 349.650.000 VND/năm |
| **Q2** | Thôi học trong 2 tuần đầu hoàn trả bao nhiêu %? | `quy-dinh-tai-chinh-bieu-phi#24` [student] — mục *Bảo lưu và hoàn trả học phí* | 0,5089 | **2 / 2** | Có — hoàn 50% |
| **Q3** | Học bổng 100% GPA 2,8 có bị hạ không? *(filter `audience=student`)* | `duy-tri-hoc-bong-ho-tro-tai-chinh#2` [student] — mục *Học bổng 100%* | 0,3360 | **2 / 2** | Có — duy trì có điều kiện, gia hạn 1 kỳ |
| **Q4** | Có những chính sách ưu đãi hoặc chiết khấu nào? | `quy-dinh-tai-chinh-bieu-phi#21` [student] — mục *Ưu đãi & chiết khấu* | 0,4394 | **2 / 2** | Có — 2,5%, 10% và 5% |
| **Q5** | Hạn nộp hồ sơ hỗ trợ tài chính kỳ Thu là khi nào? *(filter `audience=student`)* | `duy-tri-hoc-bong-ho-tro-tai-chinh#1` [student] | 0,3853 | **0 / 2** | Không — tài liệu đúng bằng tiếng Anh không vào Top-3 |

**Tổng: 8/10** — chunk thực sự chứa dữ kiện có trong Top-3 ở **4/5 câu**, MRR = **0,80**, câu trả lời đạt kiểm tra ở **4/5 câu**. Kết quả đầy đủ nằm trong `ket_qua_benchmark_thanh_vien_3.txt`.

### Thử nghiệm A/B metadata filter

- **Q3:** Có filter, văn bản chính thức `duy-tri-hoc-bong-ho-tro-tai-chinh` lên Top-1 và đạt 2/2. Không filter, ba vị trí đầu đều là FAQ `audience=all`, câu trả lời bị kéo về quy tắc chung “giảm một bậc”, nên điểm giảm còn 0/2.
- **Q5:** Không filter, trang dành cho tân sinh viên `audience=all` lên Top-1 với hạn ngày 15 sai đối tượng. Có filter đã loại bẫy này, nhưng dense lexical retrieval vẫn không vượt qua khoảng cách Việt–Anh để tìm văn bản GDL-FAO-001; vì vậy cả hai lượt đều 0/2.

Kết luận: metadata filter làm tăng độ chính xác rõ rệt ở Q3 và ngăn dùng sai chính sách ở Q5, nhưng filter không thể bù cho thiếu hụt truy xuất đa ngôn ngữ.

### Failure case

Q5 là lỗi chính. Câu hỏi viết bằng tiếng Việt, trong khi bằng chứng chính xác “20 June – 10 July / July 10th” nằm trong tài liệu tiếng Anh. `structure_tree` bảo toàn đúng mục và bảng thời gian, nhưng dense lexical baseline không đưa tài liệu đó vào Top-3. Hướng cải thiện phù hợp là giữ nguyên chunker R3 rồi bổ sung HyDE/dịch truy vấn Việt–Anh hoặc reranker đa ngôn ngữ; không nên nới gold answer thành câu chung chung “xét trong tháng 7” vì câu đó chưa trả lời được hạn 10/7.

### So sánh chiến lược của tôi với các thành viên khác trong nhóm:
1. **So với `fixed_size`:** Cùng lần chạy offline, `structure_tree` đạt 8/10 so với 6/10. Breadcrumb và việc giữ nguyên bảng/mục giúp Q2–Q4 không mất quan hệ giữa điều kiện và con số.
2. **So với `by_sentences`:** `structure_tree` đạt 8/10 so với 5/10. Sentence chunking tạo 146 chunks có độ dài dao động 11–1.772 ký tự, trong khi cấu trúc cây chỉ tạo 70 chunks và giữ được ngữ cảnh phân cấp.
3. **Bài học rút ra:** R3 là chiến lược chunking tốt nhất trong các baseline của lần chạy này, nhưng chunking tốt không tự giải quyết được truy vấn đa ngôn ngữ. Q5 cần kết hợp đóng góp hybrid/HyDE của R2; hai kỹ thuật bổ sung cho nhau thay vì thay thế nhau.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Hạng mục | Điểm tối đa | Điểm tự đánh giá |
|:---|:---:|:---:|
| Khởi động (Warm-up) | 5 | 5 / 5 |
| Hướng tiếp cận (My Approach) | 10 | 10 / 10 |
| Hoàn thiện Code (Core Implementation - 42 tests) | 30 | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 | 8 / 10 |
| **Tổng điểm phần cá nhân** | **60** | **58 / 60** |
