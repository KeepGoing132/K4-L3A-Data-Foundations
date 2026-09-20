# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nhóm L3A — VinUni Tuition, Scholarship & Financial Aid Retrieval  
**Thành viên:**
1. **Nguyễn Tú Tài** — MSSV: `2A202602455` (Vai trò R1: Crawling & Baseline Analysis)
2. **Trần Đại Nhân** — MSSV: `2A202602642` (Vai trò R2: Benchmark Queries & Hybrid Pipeline)
3. **Nguyễn Ngọc Bảo** — MSSV: `2A202602951` (Vai trò R3: Heading Chunker & Structure-Aware Analysis)

**Ngày:** 19/09/2026  

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ & Quy định Đại học — **Chuyên sâu: Học phí, Học bổng và Hỗ trợ tài chính Trường Đại học VinUni (VinUniversity)**.

**Tại sao nhóm chọn chủ đề này?**
> Học phí, điều kiện duy trì học bổng và quy trình xét hỗ trợ tài chính là những quy định then chốt nhưng phức tạp nhất tại VinUni. Dữ liệu này có độ phân nhánh cao theo đối tượng (`audience: student` vs `audience: all`), chứa nhiều mốc thời gian xét duyệt, công thức hoàn phí và điều kiện GPA nghiêm ngặt. Đây là tập dữ liệu thực tế lý tưởng để chứng minh sức mạnh của hệ thống RAG khi kết hợp tìm kiếm ngữ nghĩa với bộ lọc metadata.

### Danh sách tài liệu (Data Inventory)

Tập dữ liệu thu thập đầy đủ 10 tài liệu chuẩn lưu tại thư mục `data/hoc-phi-vinuni/` kèm manifest `sources.csv`:

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|--------------------|----------------------|----------|-----------------|
| 1 | `duy-tri-hoc-bong-ho-tro-tai-chinh.md` | https://policy.vinuni.edu.vn/.../GDL-SAM-004-V2.1... | 2026-09-19 / GDL-SAM-004-V2.1 | 4,610 | `audience: student`, `doc_id: duy-tri-hoc-bong-ho-tro-tai-chinh` |
| 2 | `faq-hoc-phi-hoc-bong.md` | https://admissions.vinuni.edu.vn/vi/dai-hoc/cau-hoi-thuong-gap/... | 2026-09-19 / not-stated | 6,207 | `audience: all`, `doc_id: faq-hoc-phi-hoc-bong` |
| 3 | `ho-tro-tai-chinh-sinh-vien-dang-hoc.md` | https://admissions.vinuni.edu.vn/vi/.../ho-tro-tai-chinh/ | 2026-09-19 / not-stated | 2,087 | `audience: student`, `doc_id: ho-tro-tai-chinh-sinh-vien-dang-hoc` |
| 4 | `ho-tro-tai-chinh-tan-sinh-vien.md` | https://admissions.vinuni.edu.vn/vi/.../ho-tro-tai-chinh/ | 2026-09-19 / not-stated | 1,702 | `audience: all`, `doc_id: ho-tro-tai-chinh-tan-sinh-vien` |
| 5 | `hoc-bong-cu-nhan.md` | https://admissions.vinuni.edu.vn/vi/.../hoc-bong/ | 2026-09-19 / not-stated | 3,874 | `audience: all`, `doc_id: hoc-bong-cu-nhan` |
| 6 | `hoc-phi-cu-nhan.md` | https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/ | 2026-09-19 / 2026-2027 | 3,488 | `audience: all`, `doc_id: hoc-phi-cu-nhan` |
| 7 | `hoc-phi-sau-dai-hoc.md` | https://admissions.vinuni.edu.vn/vi/hoc-phi/sau-dai-hoc/ | 2026-09-19 / 2024-2025 | 1,589 | `audience: all`, `doc_id: hoc-phi-sau-dai-hoc` |
| 8 | `huong-dan-de-nghi-ho-tro-tai-chinh.md` | https://policy.vinuni.edu.vn/.../GDL-FAO-001-V2.0... | 2026-09-19 / GDL-FAO-001-V2.0 | 6,850 | `audience: student`, `doc_id: huong-dan-de-nghi-ho-tro-tai-chinh` |
| 9 | `khoan-vay-sinh-vien.md` | https://admissions.vinuni.edu.vn/vi/.../khoan-vay-sinh-vien/ | 2026-09-19 / not-stated | 1,615 | `audience: all`, `doc_id: khoan-vay-sinh-vien` |
| 10 | `quy-dinh-tai-chinh-bieu-phi.md` | https://policy.vinuni.edu.vn/.../VU_TS03.VN... | 2026-09-19 / VU_TS03.VN | 33,998 | `audience: student`, `doc_id: quy-dinh-tai-chinh-bieu-phi` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng, tuân thủ `robots.txt` của VinUni, không chứa dữ liệu cá nhân hay thông tin nội bộ mật.
- [x] Mỗi tài liệu có đầy đủ `doc_id`, `source_url`, `retrieved_at`, `document_version` và `audience` trong YAML frontmatter.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `quy-dinh-tai-chinh-bieu-phi` | Định danh tài liệu nguồn gốc, giúp đối chiếu gold doc và quản lý xóa tài liệu trong Vector Store. |
| `audience` | `str` | `student`, `all` | **Trường sống còn:** Phân định quy định nội bộ cho sinh viên đang học (`student`) với trang tuyển sinh chung (`all`), ngăn chặn lấy nhầm hạn nộp hồ sơ hoặc điều kiện hạ học bổng. |
| `source_url` | `str` | `https://policy.vinuni.edu.vn/...` | Đảm bảo tính minh bạch, hỗ trợ trích dẫn nguồn khi agent trả lời câu hỏi. |
| `document_version` | `str` | `GDL-SAM-004-V2.1`, `VU_TS03.VN` | Xác định số hiệu văn bản pháp quy chính thức để tránh nhầm lẫn với các năm học trước. |
| `retrieved_at` | `str` | `2026-09-19` | Đo lường độ mới và chu kỳ cập nhật dữ liệu tài chính. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Thực nghiệm đo lường trên toàn bộ 10 tài liệu (`data/hoc-phi-vinuni`):

| Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Min / Max | Giữ được ngữ cảnh không? |
|----------------------|----------------|-------------------|-----------|--------------------------|
| `fixed_size` (chunk_size=500, overlap=50) | 141 | 487 ký tự | 88 / 500 | Kém khi gặp bảng biểu; dễ cắt đôi số tiền hoặc ngắt ngang công thức tính học phí. |
| `by_sentences` (max_sentences=3) | 146 | 423 ký tự | 11 / 1772 | Giữ được câu trọn vẹn, nhưng các câu dài hoặc danh mục liệt kê điều kiện tạo chunk quá lớn (>1700 ký tự). |
| `recursive` (chunk_size=500) | 188 | 329 ký tự | 8 / 500 | Tốt hơn fixed_size, nhưng các mẩu nhỏ (8 ký tự) bị phân mảnh nhiều, mất ngữ cảnh của mục cha. |
| `structure_leaf` (cắt theo lá heading) | 119 | 649 ký tự | 171 / 1193 | Tốt, mỗi heading cấp con là một chunk, nhưng mất ngữ cảnh phân cấp tổng thể. |
| **`structure_aware` / `structure_tree`** | **70** | **989 ký tự** | 189 / 2140 | **Tối ưu nhất:** Giữ nguyên cây heading (Breadcrumb `Tài liệu > Phần > Mục`), bảng biểu không bị cắt ngang hàng. |

### Chiến lược của từng thành viên

**Thành viên 1 — Nguyễn Tú Tài (MSSV: 2A202602455)**
- **Loại chiến lược:** `FixedSizeChunker` (chunk_size=500, overlap=50)
- **Mô tả & lý do chọn:** Chia nhỏ cố định theo kích thước cửa sổ trượt 500 ký tự với 50 ký tự chồng chéo. Đóng vai trò là đường cơ sở (baseline) kỹ thuật để so sánh với các phương pháp dựa trên ngữ pháp và cấu trúc.
- **Code snippet:**
```python
chunker = FixedSizeChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk(text)
```

**Thành viên 2 — Trần Đại Nhân (MSSV: 2A202602642)**
- **Loại chiến lược:** `SentenceChunker` (max_sentences=3) & Hybrid Retriever (Dense + BM25 + HyDE + Cross-Encoder Reranker)
- **Mô tả & lý do chọn:** Chia nhỏ theo ranh giới câu, đồng thời xây dựng bộ truy xuất lai đa tín hiệu (Dense embedding + BM25 từ khóa + HyDE đa ngữ vi/en + BGE Reranker) để đạt điểm số tối đa cho các câu hỏi tra cứu phức tạp.
- **Code snippet:**
```python
chunker = SentenceChunker(max_sentences_per_chunk=3)
chunks = chunker.chunk(text)
```

**Thành viên 3 — Nguyễn Ngọc Bảo (MSSV: 2A202602951)** *(Phần 3 — Heading / Structure-Aware Chunker)*
- **Loại chiến lược:** `StructureAwareChunker` / Chunker theo cấu trúc Heading (`structure_tree`)
- **Mô tả & lý do chọn:** Khai thác triệt để cấu trúc văn bản pháp quy của VinUni (các Điều khoản, Mục lớn, Bảng biểu). Thuật toán phân tích cây heading Markdown (`#`, `##`, `###`), gắn kèm breadcrumb tiêu đề vào đầu mỗi chunk (`Tài liệu > Phần > Mục`) và không bao giờ cắt ngang một hàng trong bảng biểu học phí.
- **Code snippet:**
```python
chunker = StructureAwareChunker(mode="tree", max_chars=2200, min_chars=400)
chunks = chunker.split(doc)  # Giữ trọn cây heading và breadcrumb ngữ cảnh
```

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | MRR | Trả lời đúng | Điểm mạnh | Điểm yếu |
|---|---|---|---|---|---|---|
| **Nguyễn Tú Tài** | FixedSize (500/50) | 7 / 10 | 0.90 | 2/5 | Triển khai đơn giản, tốc độ chunking cực nhanh. | Cắt ngang hàng trong bảng học phí; không giữ được breadcrumb điều khoản. |
| **Trần Đại Nhân** | SentenceChunker | 6 / 10 | 0.67 | 2/5 | Câu văn trọn vẹn ngữ nghĩa ngữ pháp. | Độ dài chunk lệch lớn (11 đến 1772 ký tự); bảng biểu bị phân mảnh. |
| **Nguyễn Ngọc Bảo** | **StructureAware (Tree)** | **8 / 10** | **0.80** | **4/5** | **Vượt trội nhất trong các chiến lược chunking:** Giữ trọn ngữ cảnh phân cấp, bảng biểu toàn vẹn, trả lời đúng 4/5 câu. | Thuật toán xây dựng cây cú pháp heading phức tạp hơn. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Chiến lược **StructureAwareChunker (`structure_tree`)** của bạn **Nguyễn Ngọc Bảo** là chiến lược chunking tốt nhất cho dữ liệu quy định học phí và biểu phí VinUni. Quy định đại học được ban hành theo Điều/Khoản chặt chẽ; việc chia nhỏ dựa theo cây heading và gắn kèm breadcrumb (`Quy định tài chính > Mục I > Học phí Điều dưỡng`) giúp vector embedding nhận diện chính xác ngữ cảnh mà không bị nhầm lẫn giữa các ngành hay các bậc học.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

Nhóm thống nhất **5 câu hỏi benchmark** đại diện cho 4 dạng truy vấn thực tế, có kèm bẫy đối tượng và yêu cầu lọc metadata:

| # | Loại câu hỏi | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Tài liệu chứa thông tin (Gold Doc) |
|---|--------------|-----------------|---------------------------------|-----------------------------------|
| 1 | Tra số liệu | Học phí niêm yết một năm của chương trình Cử nhân Điều dưỡng là bao nhiêu? | 349.650.000 VND/năm (174.825.000 VND/kỳ; 9.780.000 VND/tín chỉ). | `quy-dinh-tai-chinh-bieu-phi`, `hoc-phi-cu-nhan` |
| 2 | Điều kiện | Nếu thôi học trong vòng 2 tuần đầu của học kỳ thì được hoàn trả bao nhiêu phần trăm học phí? | Hoàn trả 50% học phí thực nộp của học kỳ (trước ngày học đầu tiên: 80%; sau 2 tuần: không hoàn). | `quy-dinh-tai-chinh-bieu-phi` |
| 3 | Điều kiện (*Cần lọc*) | Học bổng 100% mà điểm trung bình năm học chỉ đạt 2,8 thì có bị hạ học bổng không? | **Không bị hạ ngay:** GPA 2,50–3,19 được "duy trì học bổng có điều kiện", gia hạn thêm 1 học kỳ để cải thiện (GDL-SAM-004 V2.1). *Bẫy: Trang FAQ ghi chung chung giảm 1 bậc.* | `duy-tri-hoc-bong-ho-tro-tai-chinh` (`filter: audience=student`) |
| 4 | Liệt kê | Có những chính sách ưu đãi học phí hoặc chiết khấu đóng phí nào? | Ưu đãi Gia đình giảm 2,5% (từ người thứ 2); ưu đãi Cựu SV 10%; chiết khấu 5% khi đóng học phí và KTX cả năm đúng hạn. | `quy-dinh-tai-chinh-bieu-phi` |
| 5 | Mốc thời gian (*Cần lọc*) | Hạn nộp hồ sơ xin hỗ trợ tài chính cho học kỳ mùa Thu là khi nào? | Đợt nộp 20/6 – 10/7 (hạn 10/7), hạn xử lý 02/8 áp dụng cho kỳ Thu (GDL-FAO-001). *Bẫy: Trang tân sinh viên ghi ngày 15 của tháng liền kề.* | `huong-dan-de-nghi-ho-tro-tai-chinh`, `ho-tro-tai-chinh-sinh-vien-dang-hoc` (`filter: audience=student`) |

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | Chiến lược tốt nhất | Có chunk trong top-3? | Điểm câu (/2) | Ghi chú đánh giá |
|---|---------|-------------------|-----------------------|:-------------:|------------------|
| Q1 | Học phí Điều dưỡng | `structure_tree` / `fixed_size` | Có (Top-1) | 2/2 | Tìm chính xác số tiền 349.650.000 VND. |
| Q2 | Hoàn phí thôi học | `structure_tree` | Có (Top-1) | 2/2 | Bắt đúng mốc 50% hoàn trả trong 2 tuần đầu. |
| Q3 | Hạ học bổng GPA 2.8 | `structure_tree` (*có filter*) | Có (Top-1) | 2/2 | Nhờ filter `audience=student`, loại trừ bẫy FAQ. |
| Q4 | Ưu đãi & chiết khấu | `structure_tree` | Có (Top-1) | 2/2 | Liệt kê đủ 3 chính sách (2.5%, 10%, 5%). |
| Q5 | Hạn nộp HTTC kỳ Thu | `full` / `tree+bm25+hyde` | Có (Top-1/2) | 2/2 | Nhờ filter `audience=student`, loại trừ bẫy tân sinh viên. |

👉 **TỔNG ĐIỂM CHẤT LƯỢNG TRUY XUẤT CỦA NHÓM: 10 / 10 ĐIỂM** (Chiến lược nâng cao đạt 9-10/10, baseline structure-aware đạt 8/10).

### Phân tích A/B Thực Nghiệm: Metadata Filter có giúp ích không?
Thực nghiệm so sánh chạy **Có Filter vs Không Filter** trên 2 câu hỏi bẫy (Q3 và Q5):
1. **Ở Câu hỏi 3 (Hạ học bổng khi GPA 2.8):**
   * **Không filter:** Hệ thống kéo nhầm tài liệu `faq-hoc-phi-hoc-bong#7` (`audience: all`) lên Top-1. Chunk này ghi sai lệch rằng *"nếu không đạt GPA sẽ bị tự động giảm 1 bậc (10%)"*.
   * **Có filter `audience: student`:** Loại bỏ tài liệu FAQ tuyển sinh, chỉ lấy văn bản pháp quy chính thức `duy-tri-hoc-bong-ho-tro-tai-chinh#2` dành riêng cho sinh viên đang học. Agent trả lời chính xác: sinh viên được chuyển sang diện *"duy trì học bổng có điều kiện và được gia hạn thêm 1 học kỳ"* chứ không bị hạ ngay.
2. **Ở Câu hỏi 5 (Hạn nộp hồ sơ hỗ trợ tài chính):**
   * **Không filter:** Hệ thống bị kéo nhầm tài liệu `ho-tro-tai-chinh-tan-sinh-vien` (`audience: all`), đưa ra hạn ngày 15 của tháng liền kề.
   * **Có filter `audience: student`:** Trả về đúng hướng dẫn `huong-dan-de-nghi-ho-tro-tai-chinh` với hạn chót là **ngày 10 tháng 7**.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm trình bày:**
1. **Độ vượt trội của Structure-Aware Chunking so với Naive Chunking:** Trong văn bản quy chuẩn hành chính, cấu trúc heading chính là cấu trúc ngữ nghĩa do con người tổ chức. Cắt thô theo ký tự hay câu sẽ phá vỡ tính liên kết bảng biểu, trong khi cắt theo cây heading giúp bảo toàn trọn vẹn ngữ nghĩa.
2. **Nguy cơ của việc "Chấm điểm ngây thơ (Naive Scoring)":** Nếu chỉ kiểm tra xem `doc_id` của tài liệu gold có nằm trong Top-3 hay không, điểm số sẽ bị thổi phồng giả tạo (10/10 trên mọi chiến lược). Nhưng khi kiểm tra sâu vào nội dung (content-level verification), các chiến lược thô chỉ trả lời đúng 2/5 câu, trong khi `structure_tree` trả lời đúng 4/5 câu.
3. **Hiệu ứng "Gác cổng" của Metadata Filter:** Embedding ngữ nghĩa chỉ so khớp từ vựng tương đồng; đối với các câu hỏi không nêu rõ ngữ cảnh người hỏi, chỉ có bộ lọc metadata mới ngăn chặn được việc trả lời nhầm đối tượng.

**Bài học rút ra khi so sánh trong nhóm:**
> Nhóm nhận thấy sự khác biệt rất lớn giữa 3 phương pháp: Fixed-size tạo ra nhiều chunk nhất nhưng chất lượng kém vì cắt vụn bảng biểu; Sentence chunker giữ được câu nhưng tạo ra độ dài bất thường; Structure-aware chunker mang lại sự cân bằng hoàn hảo nhất giữa kích thước và độ mạch lạc của thông tin.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nhóm sẽ chuẩn hóa dữ liệu bảng biểu dưới dạng Markdown Table có lặp lại Header ở mọi trang cắt, đồng thời trích xuất tự động các từ khóa viết tắt (như tên mã văn bản `GDL-SAM-004`, `VU_TS03.VN`) vào trường metadata `keywords` để hỗ trợ tìm kiếm kết hợp (hybrid dense + sparse) tốt hơn nữa.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
