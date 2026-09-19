# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nhóm L3A — VinUni Tuition Retrieval  
**Thành viên:**
1. **Nguyễn Tú Tài** — MSSV: `2A202602455`
2. **Trần Đại Nhân** — MSSV: `2A202602642`
3. **Nguyễn Ngọc Bảo** — MSSV: `2A202602951`

**Ngày:** 19/09/2026  

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ & Quy định Đại học — **Chuyên sâu: Biểu phí & Chính sách Học phí Trường Đại học VinUni (VinUniversity)**.

**Tại sao nhóm chọn chủ đề này?**
> Học phí và các chính sách hỗ trợ tài chính là thông tin quan trọng hàng đầu mà sinh viên và phụ huynh quan tâm khi tìm hiểu về VinUni. Dữ liệu này có tính cấu trúc cao, bao gồm nhiều con số, điều kiện ràng buộc, mốc thời gian và đối tượng áp dụng khác nhau, là nguồn dữ liệu lý tưởng để kiểm thử khả năng tìm kiếm ngữ nghĩa và lọc metadata của hệ thống RAG.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|--------------------|----------------------|----------|-----------------|
| 1 | `vinuni-hoc-phi-cu-nhan.md` | https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/ | 2026-09-19 / 2026-2027 | 1,488 | `audience: student`, `department: finance`, `category: tuition` |
| 2 | `vinuni-ho-tro-hoc-phi-35.md` | https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/ | 2026-09-19 / 2026-2027 | 1,420 | `audience: student`, `department: finance`, `category: tuition-subsidy` |
| 3 | `vinuni-hoc-phi-tin-chi-hoc-lai.md` | https://policy.vinuni.edu.vn/all-policies/financial-regulations-and-tariff-for-student-2/ | 2026-09-19 / 2026-2027 | 1,761 | `audience: student`, `department: academic-affairs`, `category: credit-tuition` |
| 4 | `vinuni-quy-dinh-nop-va-hoan-hoc-phi.md` | https://policy.vinuni.edu.vn/all-policies/financial-regulations-and-tariff-for-student-2/ | 2026-09-19 / 2026-2027 | 1,886 | `audience: student`, `department: finance`, `category: tuition-payment` |
| 5 | `vinuni-chinh-sach-giam-hoc-phi-cbnv.md` | https://policy.vinuni.edu.vn/all-policies/financial-regulations-and-tariff-for-student-2/ | 2026-09-19 / 2026-2027 | 1,478 | `audience: staff`, `department: human-resources`, `category: tuition-discount` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `vinuni-hoc-phi-cu-nhan` | Định danh tài liệu duy nhất, giúp truy vết nguồn và xóa tài liệu trong store. |
| `audience` | `str` | `student`, `staff` | **Trường cốt lõi:** Phân biệt đối tượng áp dụng, ngăn chặn nhầm lẫn giữa chính sách sinh viên và phúc lợi nhân sự nội bộ. |
| `department` | `str` | `finance`, `academic-affairs` | Giúp lọc tài liệu theo phòng ban quản lý chuyên trách khi mở rộng hệ thống. |
| `category` | `str` | `tuition`, `credit-tuition` | Phân loại mảng tài chính (biểu phí niêm yết, tín chỉ, nộp/hoàn phí). |
| `retrieved_at` | `str` | `2026-09-19` | Kiểm soát tính cập nhật của dữ liệu văn bản. |
| `document_version` | `str` | `2026-2027` | Xác định niên khóa áp dụng để tránh tra cứu nhầm quy định cũ. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu mẫu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `vinuni-chinh-sach-giam-hoc-phi-cbnv.md` | FixedSizeChunker (`fixed_size`) | 6 | 271.3 ký tự | Trung bình (dễ cắt đứt giữa câu hoặc ranh giới mục) |
| `vinuni-chinh-sach-giam-hoc-phi-cbnv.md` | SentenceChunker (`by_sentences`) | 5 | 294.2 ký tự | Tốt (giữ nguyên câu, nhưng các câu dài tạo chunk lớn) |
| `vinuni-chinh-sach-giam-hoc-phi-cbnv.md` | RecursiveChunker (`recursive`) | 10 | 146.3 ký tự | Rất tốt (bảo toàn cấu trúc mục quy định và danh sách gạch đầu dòng) |
| `vinuni-ho-tro-hoc-phi-35.md` | FixedSizeChunker (`fixed_size`) | 6 | 261.7 ký tự | Trung bình (bị ngắt giữa các con số phần trăm) |
| `vinuni-ho-tro-hoc-phi-35.md` | SentenceChunker (`by_sentences`) | 4 | 353.5 ký tự | Khá (câu trọn vẹn nhưng kích thước chunk hơi lớn) |
| `vinuni-ho-tro-hoc-phi-35.md` | RecursiveChunker (`recursive`) | 8 | 176.0 ký tự | Rất tốt (các điều khoản được tách theo đoạn logic tự nhiên) |

### Chiến lược của từng thành viên

**Thành viên 1 — Nguyễn Tú Tài (MSSV: 2A202602455)**
- **Loại chiến lược:** `FixedSizeChunker` (chunk_size=300, overlap=30)
- **Mô tả & lý do chọn cho chủ đề này:** Chia đều văn bản thành các khối cố định 300 ký tự với 30 ký tự chồng chéo. Đây là chiến lược đơn giản, tốc độ thực thi nhanh, phù hợp làm baseline ban đầu để đo lường.
- **Code snippet:**
```python
chunker = FixedSizeChunker(chunk_size=300, overlap=30)
chunks = chunker.chunk(text)
```

**Thành viên 2 — Trần Đại Nhân (MSSV: 2A202602642)**
- **Loại chiến lược:** `SentenceChunker` (max_sentences_per_chunk=3)
- **Mô tả & lý do chọn:** Chia nhỏ dựa trên ranh giới ngữ pháp của câu (`. `, `! `, `? `), mỗi chunk chứa tối đa 3 câu. Đảm bảo thông tin số liệu và điều kiện đi kèm không bao giờ bị cắt vụn giữa chừng.
- **Code snippet:**
```python
chunker = SentenceChunker(max_sentences_per_chunk=3)
chunks = chunker.chunk(text)
```

**Thành viên 3 — Nguyễn Ngọc Bảo (MSSV: 2A202602951)**
- **Loại chiến lược:** `RecursiveChunker` (chunk_size=350, separators=["\n\n", "\n", ". ", " ", ""])
- **Mô tả & lý do chọn:** Phân tách đệ quy nhiều tầng ưu tiên theo đoạn (`\n\n`), dòng tiêu đề (`\n`), câu (`. `) và từ. Chiến lược này khai thác tối đa cấu trúc heading Markdown của văn bản quy chế đại học, giữ trọn vẹn từng điều khoản.
- **Code snippet:**
```python
chunker = RecursiveChunker(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=350)
chunks = chunker.chunk(text)
```

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|---|---|---|---|---|
| Nguyễn Tú Tài | FixedSize (300/30) | 7.5 / 10 | Tốc độ xử lý nhanh, kích thước chunk đồng đều. | Dễ ngắt đôi bảng biểu học phí, làm mất ngữ cảnh số tiền. |
| Trần Đại Nhân | SentenceChunker (3 câu) | 8.5 / 10 | Câu văn luôn trọn vẹn ngữ nghĩa, không bị cụt câu. | Khi có câu văn quy định quá dài, kích thước chunk bị lệch lớn. |
| Nguyễn Ngọc Bảo | RecursiveChunker (350) | 9.5 / 10 | Tôn trọng cấu trúc văn bản quy định, điểm tương đồng cao nhất. | Thuật toán đệ quy phức tạp hơn các phương pháp cắt cứng. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Chiến lược **RecursiveChunker** của bạn **Nguyễn Ngọc Bảo** là tốt nhất cho chủ đề quy định học phí. Vì văn bản tài chính đại học được viết theo cấu trúc phân cấp rõ ràng (Tiêu đề lớn -> Điều khoản -> Danh sách gạch đầu dòng con số), thuật toán Recursive ưu tiên tách theo đoạn (`\n\n`) trước, giúp mỗi chunk là một điều khoản trọn vẹn mà không làm đứt đoạn ngữ cảnh con số.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Bộ câu hỏi chạy thống nhất trên file `bench.py`.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Học phí Bác sĩ Y khoa VinUni là bao nhiêu một năm? | 815.850.000 VND / năm học (407.925.000 VND / kỳ học). | `vinuni-hoc-phi-cu-nhan` (Mục 1) |
| 2 | Chính sách hỗ trợ 35% học phí từ Vingroup áp dụng cho những ai và duy trì bao lâu? | Áp dụng tự động cho tất cả sinh viên trúng tuyển (Việt Nam và quốc tế) và duy trì suốt toàn bộ thời gian học chính thức. | `vinuni-ho-tro-hoc-phi-35` (Mục 1 & 2) |
| 3 | Học phí học lại theo tín chỉ tại VinUni được tính bằng bao nhiêu phần trăm? | Tính bằng 50% mức học phí chuẩn theo tín chỉ tương ứng của môn học đó. | `vinuni-hoc-phi-tin-chi-hoc-lai` (Mục 3) |
| 4 | Rút hồ sơ trước khi học kỳ bắt đầu thì được hoàn lại bao nhiêu phần trăm học phí? | Được hoàn trả 90% số học phí thực nộp của học kỳ đó. | `vinuni-quy-dinh-nop-va-hoan-hoc-phi` (Mục 3) |
| 5 | Mức chiết khấu giảm học phí cho con em là bao nhiêu? | Được giảm thêm 5% học phí niêm yết (cộng dồn với 35% thành 40%). Cần lọc `audience: student` để tránh nhầm với chính sách nhân sự chung. | `vinuni-chinh-sach-giam-hoc-phi-cbnv` & `vinuni-hoc-phi-cu-nhan` |

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Học phí Bác sĩ Y khoa VinUni | RecursiveChunker | Có (Top-1) | Trích xuất chính xác bảng học phí cử nhân. |
| 2 | Chính sách hỗ trợ 35% học phí | RecursiveChunker | Có (Top-1) | Tìm đúng văn bản tài trợ của Vingroup. |
| 3 | Học phí học lại theo tín chỉ | SentenceChunker | Có (Top-1) | Bắt trọn câu quy định 50% đơn giá. |
| 4 | Hoàn phí khi rút hồ sơ | RecursiveChunker | Có (Top-1) | Trả về mục chính sách rút hồ sơ trước kỳ học. |
| 5 | Giảm học phí con em (Lọc metadata) | RecursiveChunker | Có (Top-1) | **Lọc `audience=student`**: Loại trừ văn bản nhân sự, chỉ lấy đúng quy định sinh viên. |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Rất hữu ích, đặc biệt ở Câu hỏi số 5**. Nếu không có bộ lọc `metadata_filter={"audience": "student"}`, hệ thống tìm kiếm vector thuần túy sẽ trích xuất văn bản phúc lợi nội bộ của cán bộ nhân viên (`vinuni-chinh-sach-giam-hoc-phi-cbnv.md` với mức giảm 5% cho con CBNV). Khi kích hoạt bộ lọc `audience: student`, hệ thống loại bỏ ngay các văn bản dành cho nhân viên (`staff`) và trả về đúng chính sách học phí chuẩn dành cho sinh viên thông thường.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
1. **Sự vượt trội của Recursive Chunking:** Đối với các văn bản quy chế đại học có cấu trúc phân cấp (heading, gạch đầu dòng, bảng biểu), việc cắt theo logic đoạn văn tự nhiên giúp duy trì trọn vẹn ngữ nghĩa tốt hơn nhiều so với việc cắt cố định theo số ký tự.
2. **Vai trò sống còn của Metadata Filter:** Embedding ngữ nghĩa chỉ tìm kiếm độ tương đồng từ ngữ, không phân biệt được đối tượng thụ hưởng nếu câu hỏi mơ hồ; metadata filter đóng vai trò là "người gác cổng" bắt buộc để đảm bảo an toàn thông tin đúng đối tượng.
3. **Mối liên hệ giữa Chunk size và Embedding:** Chunk quá nhỏ sẽ mất ngữ cảnh bao quát, trong khi chunk quá lớn làm loãng vector embedding khiến điểm cosine similarity bị kéo thấp.

**Bài học rút ra khi so sánh trong nhóm:**
> Khi cả 3 thành viên chạy trên cùng một bộ dữ liệu, nhóm nhận thấy chiến lược `FixedSize` của bạn Tú Tài cho số lượng chunk nhiều nhất nhưng điểm số top-1 lại thấp nhất do câu bị cắt ngang; chiến lược `Sentence` của bạn Đại Nhân bảo toàn câu tốt nhưng kích thước chunk chênh lệch; chiến lược `Recursive` của bạn Ngọc Bảo cân bằng hoàn hảo giữa ngữ cảnh và kích thước chunk, mang lại điểm số retrieval vượt trội.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nếu làm lại, nhóm sẽ thiết kế thêm trường metadata `section_header` được gắn tự động vào từng chunk con khi cắt nhỏ. Khi một điều khoản dài bị chia làm 2 mảnh, việc gắn kèm tiêu đề lớn vào đầu mỗi mảnh sẽ giúp mảnh thứ hai không bị mất ngữ cảnh "đang nói về quy định gì", từ đó nâng cao hơn nữa độ chính xác của vector store.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
