# Giai đoạn B — B1: baseline trên đặc trưng user-day (CERT r4.2 và r6.2)

Đặc tả và pre-registration: `docs/phaseB_spec.md` (B0 + mục **"B0 v2 — 2026-10-16"**, kèm "Chi tiết thực thi" 1–16 chốt trước khi chạy). Giai đoạn A: `REPORT.md`.
Ngày chạy: 2026-10-17. Môi trường: Python 3.12, polars 2.0.0, pyarrow 21.0.0, scikit-learn 1.7.2, numpy 2.5.3 (không cài gói mới). Bảng đầy đủ: `results/phaseB_tables.md`. **B1 được duyệt 2026-10-17. Đang dừng ở cổng B2-0, chưa viết code mô hình B2.**

**Sửa sau khi B1 được duyệt (2026-10-17, không chạy lại gì, không đổi số):**
- (a) câu hỏi 3: ghi rõ cơ cấu kịch bản của test PB-T r4.2; kết luận "giữ được" chỉ áp dụng cho kịch bản 2 (§1, §6.5 mục 5, §8);
- (b) test (e) được ghi là kiểm tra thiết kế, không phải kiểm tra thực nghiệm (§1, §7);
- (c) thêm giới hạn cho phân tích đơn biến `z_g8_http` (§1, §6.5 mục 2, §7).

---

## 1. Tóm tắt

- **Đã làm:**
  - Trích đặc trưng user-day cho cả hai bản, khớp từng user-day với giai đoạn A: r4.2 330 452 user-day × 95 cột; r6.2 1 394 010 × 105 cột.
  - Pass 5 test (a)–(e). Toàn bộ test: 25/25. Lưu ý: test (e) là **kiểm tra thiết kế** (chữ ký hàm trích đặc trưng không nhận nhãn hay cách chia; chạy lại cho kết quả giống hệt), **không phải** kiểm tra thực nghiệm (xem §7).
  - Chạy mọi baseline §7 cùng hai tham chiếu (có giám sát; có thông tin kịch bản) trên PB-U (10 fold) và PB-T (1 lần chia × 5 seed) cho r4.2 và r6.2.
  - Có đủ các bảng bắt buộc V2.1, V2.4, V2.5, V2.9, V2.10, V2.11.
- **Câu hỏi 1 (đặc trưng user-day có vượt xa giai đoạn A không?):** baseline **không giám sát** thì **không**; đặc trưng thì **có chứa** nhiều thông tin.
  - Baseline không giám sát tốt nhất ở PB-U r4.2 là **z-score, tổng z dương**: AUC 0.81, PR-AUC **0.042**, recall@1% 15%. Isolation Forest đạt AUC 0.82, PR-AUC 0.019.
  - Mức này ngang P3 user-day của giai đoạn A (AUC 0.63–0.83, PR-AUC 0.008–0.040), không vượt xa.
  - Tham chiếu **có giám sát** (HGB, tham số mặc định) trên cùng đặc trưng đạt PR-AUC **0.59** và recall@1% **84%** (tỉ lệ dương 0.41%).
  - Phân tích đơn biến làm **sau** khi xem kết quả, chỉ trên tập fit: một đặc trưng D duy nhất, `z_g8_http` (lượt http lệch so với lịch sử của chính user), đã có PR-AUC 0.51. Phần lớn đến từ kịch bản 2. **Giới hạn:** tín hiệu này có thể một phần là dấu vết của cách CERT chèn hoạt động độc hại vào dữ liệu tổng hợp; chưa suy rộng cho dữ liệu thật.
- **Mức insider dễ hơn nhiều so với mức user-day.** Với ngân sách 1% user-day, z-score tổng z dương phát hiện **94%** insider test của r4.2 PB-U (≥ 1 ngày dương lọt top 1%), dù recall user-day chỉ 15%.
- **Câu hỏi 3 (chia theo thời gian):** ở r4.2, kết quả PB-T gần PB-U:
  - z tổng dương: PR-AUC 0.040 (PB-T) so với 0.042 (PB-U);
  - IF: 0.014 so với 0.019;
  - HGB: 0.71 so với 0.59.
  - Không insider nào có user-day dương ở cả train lẫn test PB-T, nên HGB không thể học thuộc user.
  - **Giới hạn:** test PB-T r4.2 có **333/360 ngày dương (92%) thuộc kịch bản 2**; kịch bản 1 chỉ có 25 ngày (7 insider) và kịch bản 3 chỉ có 2 ngày (1 insider). Vì vậy kết luận "giữ được" **chỉ áp dụng cho kịch bản 2**. Riêng kịch bản 2, PR-AUC PB-T so với PB-U là: z tổng dương 0.029 so với 0.026, IF 0.012 so với 0.014, HGB 0.72 so với 0.59. Kịch bản 1 và 3 có quá ít dương trong test PB-T để kết luận.
- **r6.2** (5 insider, dương 0.01–0.02%):
  - Chỉ z-score còn tín hiệu: PB-T PR-AUC 0.014, gấp ~68 lần tỉ lệ dương; cả 2 insider test lọt top 0.5%.
  - HGB thất bại (AUC 0.38–0.48), vì kịch bản trong train khác kịch bản trong test.
- **Khoá cho câu hỏi 2 (B2), theo V2.12:** baseline không giám sát tốt nhất trên PB-U r4.2 là **"z-score: tổng z dương"**, PR-AUC trung bình **0.0418**. Mô hình sâu chỉ được ghi "hơn baseline" nếu PR-AUC cao hơn baseline này ở ≥ 8/10 fold **và** trung bình tăng tương đối ≥ 10% (tức ≥ 0.0460).
- **Lỗi chính:** đặc trưng r6.2 làm sập RAM hai lần (trình đọc CSV streaming của polars giữ không giới hạn cột url của file http 90 GB). Đã sửa bằng cách đọc theo khối pyarrow 32 MB; bản r4.2 kiểm chứng giống hệt. Chi tiết ở §7.

---

## 2. Quá trình

| Bước | Việc | Lệnh / file | Thời gian |
|---|---|---|---|
| B0 v2 | Ghi 12 bổ sung + chi tiết thực thi 1–16 vào spec trước khi chạy; sửa câu §6.6 của `REPORT.md` | `docs/phaseB_spec.md`, `REPORT.md` | — |
| 1 | Trích đặc trưng r4.2 (bảng trung gian theo (user, ngày, pc / domain) → bảng user-day → lịch sử chỉ dùng quá khứ) | `python src/userday.py 4.2` | 55 s |
| 2 | Trích đặc trưng r6.2: hai lần hết RAM → đổi trình đọc http → chạy lại dưới bộ giám sát RAM 20 GB | `python src/userday.py 6.2` | 8.4 phút; RAM đỉnh 12.7 GB |
| 3 | Test (a)–(e) + toàn bộ test cũ | `pytest tests` | 39 s, **25/25 đạt** |
| 4 | Baseline B1 | `python src/baselines_b.py <rel> PBU\|PBT` | r4.2: PB-T 72 s, PB-U ~3 phút; r6.2: PB-T 4.6 phút, PB-U 8.6 phút |
| 5 | Bảng | `python src/tables_b.py` → `results/phaseB_tables.md`, điền bảng vào file này | ~1 phút |

---

## 3. Kết nối Hugging Face

Giai đoạn B **không truy cập mạng**. Mọi đặc trưng tính từ bản local đã xác minh ở giai đoạn A ("local, hash khớp HF", revision `4107ac90fefc3581abe16e1d5150fbb91d728257`, `results/hash_log.jsonl`). Nhãn lấy từ bảng đếm giai đoạn A (`data/counts`) và `answers` đã giải nén.

---

## 4. Dữ liệu, đặc trưng và nhãn

**Đặc trưng.** Lưu tại `data/userday/r<rel>.parquet`; schema và hash ở `results/phaseB/feature_schema_r<rel>.json`.

<!--TABLE:desc0-->
| Bản | user-day | cột | nhóm G | nhóm D | nhóm R | nhóm I | sha256 schema (16 ký tự đầu) | ngày |
|---|---|---|---|---|---|---|---|---|
| r4.2 | 330452 | 95 | 59 | 29 | 0 | 3 | 8f0ee716aa955a7a | 2010-01-02 → 2011-05-17 |
| r6.2 | 1394010 | 105 | 59 | 29 | 10 | 4 | f105869956c2bcba | 2010-01-02 → 2011-06-01 |
<!--/TABLE-->

- Nhóm G (59 cột):
  - 28 cột activity × khung giờ;
  - 6 cột ngoài giờ và cuối tuần;
  - 3 cột phiên làm việc;
  - 2 cột PC;
  - 2 cột USB;
  - 6 cột file;
  - 7 cột email;
  - 4 cột http và 1 cờ `khoi_dau`.
- Nhóm D: 14 z, 14 tỉ lệ và cờ `lich_su_ngan`. Nhóm R (chỉ r6.2): 10. Nhóm I: r4.2 có 3, r6.2 có 4 (V2.4).
- Spec B0 ghi "G ≈ 38 chiều" ở phần tóm tắt, nhưng bảng §3.1 cộng lại ra 58. Số thực tế là 59 vì có thêm cờ `khoi_dau` theo V2.1. Đặc trưng không đổi so với bảng §3.1.

**Nhãn.** Nhãn cửa sổ ở mức user-day, bộ insider `main`, lấy từ bảng đếm giai đoạn A. Khoá (user, ngày) của bảng đặc trưng trùng 100% với bảng đếm. Số dòng của từng user-day cũng khớp (đã assert khi nạp).

**V2.1 — cold start:** trong 30 ngày đầu không có user-day dương nào, ở cả hai bản.

<!--TABLE:desc1-->
| Bản | 30 ngày đầu | user-day | user-day dương (cửa sổ) | user-day dương (sự kiện) | insider có dương |
|---|---|---|---|---|---|
| r4.2 | 2010-01-02 → 2010-01-31 | 20549 | 0 | 0 | 0 |
| r6.2 | 2010-01-02 → 2010-01-31 | 82056 | 0 | 0 | 0 |
<!--/TABLE-->

**V2.4 — nhóm I.** Ở r4.2:
- `wikileaks` rất đặc hiệu: 69 lượt, cả 30 insider kịch bản 1 đều có.
- `job` không đặc hiệu: 472 nghìn lượt, vì chuỗi con "job" xuất hiện trong URL ngẫu nhiên.

Danh sách từ được giữ nguyên như B0.

<!--TABLE:desc2-->
| Bản | Nhóm | Dùng ở bản này | Lượt http khớp | user-day có khớp | user-day dương có khớp / tổng dương | insider có khớp trong ngày dương |
|---|---|---|---|---|---|---|
| r4.2 | wikileaks | có | 69 | 68 | 68 / 1364 | 30 |
| r4.2 | job | có | 472,178 | 98,802 | 769 / 1364 | 51 |
| r4.2 | keylog | có | 89,614 | 28,689 | 110 / 1364 | 23 |
| r4.2 | dropbox | không | 71,468 | 36,909 | 109 / 1364 | 22 |
| r6.2 | wikileaks | có | 7 | 2 | 2 / 95 | 1 |
| r6.2 | job | có | 692,897 | 244,202 | 49 / 95 | 4 |
| r6.2 | keylog | có | 462 | 438 | 1 / 95 | 1 |
| r6.2 | dropbox | có | 1,168,971 | 348,414 | 36 / 95 | 2 |
<!--/TABLE-->

**V2.5 — PB-T, kịch bản × giai đoạn** (chỉ mô tả; mốc không đổi). r4.2 PB-T có 18 insider test. Không insider nào có user-day dương ở cả train và test. Với r6.2:
- train có kịch bản 1, 3, 5 (9 user-day dương);
- val không có dương;
- test có kịch bản 2 và 4 (86 dương).

<!--TABLE:desc3-->
| Bản | Kịch bản | train: insider / dương | val: insider / dương | test: insider / dương | insider có dương ở cả train và test |
|---|---|---|---|---|---|
| r4.2 | 1 | 19 / 114 | 4 / 17 | 7 / 25 | 0 |
| r4.2 | 2 | 20 / 650 | 9 / 205 | 10 / 333 | 0 |
| r4.2 | 3 | 7 / 14 | 2 / 4 | 1 / 2 | 0 |
| r6.2 | 1 | 1 / 6 | 0 / 0 | 0 / 0 | 0 |
| r6.2 | 2 | 0 / 0 | 0 / 0 | 1 / 41 | 0 |
| r6.2 | 3 | 1 / 2 | 0 / 0 | 0 / 0 | 0 |
| r6.2 | 4 | 0 / 0 | 0 / 0 | 1 / 45 | 0 |
| r6.2 | 5 | 1 / 1 | 0 / 0 | 0 / 0 | 0 |
<!--/TABLE-->

**V2.10 — chẩn đoán "insider rời sớm"** (phân tích sau khi xem kết quả giai đoạn A). Với r4.2, giả thuyết ở `REPORT.md` §6.6 **được xác nhận**:
- insider có trung vị 214 ngày hoạt động, người thường 346;
- ngày hoạt động cuối của insider có trung vị là 2010-10-29, của người thường là 2011-05-16;
- 97% insider ngừng hoạt động hơn 30 ngày trước khi bản dữ liệu kết thúc, so với 9% người thường;
- số dòng mỗi ngày hoạt động gần như ngang nhau (TB 101 so với 99).

Vậy AUC 0.33 của "log tổng số dòng" ở giai đoạn A là do **số ngày**, không phải cường độ hoạt động. Ở r6.2 thì 2/5 insider (kịch bản 4, 5) ở lại tới cuối.

<!--TABLE:desc4-->
| Bản | Nhóm | Số user | Ngày hoạt động: TB / trung vị | Dòng / ngày hoạt động: TB / trung vị | Ngày hoạt động cuối (trung vị) | Tỉ lệ dừng > 30 ngày trước ngày cuối của bản |
|---|---|---|---|---|---|---|
| r4.2 | người thường | 930 | 338.7 / 346 | 98.9 / 105.8 | 2011-05-16 | 9% |
| r4.2 | insider (tất cả) | 70 | 220.7 / 214 | 101.4 / 112.7 | 2010-10-29 | 97% |
| r4.2 | insider kịch bản 1 | 30 | 219.9 / 218 | 94.2 / 106.4 | 2010-11-02 | 100% |
| r4.2 | insider kịch bản 2 | 30 | 231.8 / 222 | 94.1 / 112.7 | 2010-11-30 | 97% |
| r4.2 | insider kịch bản 3 | 10 | 189.9 / 179 | 145.0 / 133.1 | 2010-10-04 | 90% |
| r6.2 | người thường | 3995 | 348.6 / 356 | 96.7 / 105.9 | 2011-05-31 | 9% |
| r6.2 | insider (tất cả) | 5 | 272.2 / 316 | 147.1 / 133.5 | 2011-04-01 | 60% |
| r6.2 | insider kịch bản 1 | 1 | 176.0 / 176 | 179.2 / 179.2 | 2010-09-09 | 100% |
| r6.2 | insider kịch bản 2 | 1 | 316.0 / 316 | 196.8 / 196.8 | 2011-04-01 | 100% |
| r6.2 | insider kịch bản 3 | 1 | 157.0 / 157 | 133.5 / 133.5 | 2010-08-13 | 100% |
| r6.2 | insider kịch bản 4 | 1 | 356.0 / 356 | 106.4 / 106.4 | 2011-05-31 | 0% |
| r6.2 | insider kịch bản 5 | 1 | 356.0 / 356 | 119.6 / 119.6 | 2011-05-31 | 0% |
<!--/TABLE-->

**Kích thước tập** (PB-U r6.2: mỗi fold test có đúng 1 insider):

<!--TABLE:sizes-->
| Bản | Giao thức | Số lần chia | user-day fit | dương fit | user-day val | dương val | user-day test | dương test | insider test |
|---|---|---|---|---|---|---|---|---|---|
| r4.2 | PBU | 10 | 211292 ± 451 [210736, 211997] | 868 ± 80 [749, 1024] | 53069 ± 498 [52243, 53802] | 223 ± 41 [142, 277] | 66090 ± 352 [65615, 66875] | 273 ± 71 [135, 358] | 14 |
| r4.2 | PBT | 1 | 209182 | 778 | 37241 | 226 | 84029 | 360 | 18 |
| r6.2 | PBU | 10 | 891977 ± 1277 [890043, 894395] | 61 ± 31 [9, 92] | 223231 ± 1643 [220890, 225664] | 15 ± 18 [1, 45] | 278802 ± 1403 [276308, 281092] | 19 ± 20 [1, 45] | 1 |
| r6.2 | PBT | 1 | 843379 | 9 | 155069 | 0 | 395562 | 86 | 2 |
<!--/TABLE-->

---

## 5. Baseline và cấu hình (đúng §7 + V2.7, V2.11)

| Tên | Định nghĩa | Học gì |
|---|---|---|
| Ngẫu nhiên | giải tích: AUC 0.5, PR-AUC = tỉ lệ dương, recall@b = b | — |
| Khối lượng ± | ± log(số dòng của user-day); dấu chọn theo AUC trên tập **fit** (V2.7), báo cả hai dấu | dấu |
| Ô hiếm | tỉ lệ dòng của user-day nằm trong ô F38 hiếm (≤ 1% số dòng fit), như addendum (b) | ô hiếm, trên fit |
| Luật ¹ | log1p(connect ngoài giờ) + log1p(file ngoài giờ) + cờ USB lần đầu, cố định. **Thiết kế khi đã biết hình dạng kịch bản** | — |
| z-score | max \|z\| và tổng z dương trên 14 đặc trưng z của nhóm D | — |
| Isolation Forest | 200 cây, max_samples 256, còn lại mặc định; G+D (r6.2 thêm G+D+R); fit trên user-day fit, không dùng nhãn | không giám sát |
| *HGB (tham chiếu)* | `HistGradientBoostingClassifier` mặc định, nhãn fit; G+D (r6.2 thêm G+D+R) | có giám sát |
| *IF / HGB G+D+I* | thêm nhóm I — **"có thông tin kịch bản"** | |

- `random_state`: PB-U dùng 1000 + chỉ số lần chạy; PB-T dùng seed 0–4. Mọi tham số (`get_params()`), phiên bản sklearn/numpy, hash schema, số vòng HGB thực tế và việc early stopping có bật hay không đều được ghi trong `results/phaseB/b1_r<rel>_<proto>.json`.
- **Early stopping của HGB (V2.11):** `early_stopping='auto'`, nên bật vì train > 10 000 mẫu. sklearn tự tách 10% tập fit và dừng khi loss trên phần đó không giảm sau 10 vòng. Thực tế mô hình dừng ở **10–11 vòng** trong mọi lần chạy. Chúng tôi không dùng val hay test của mình để dừng sớm. HGB không được tinh chỉnh, nên đây là tham chiếu chứ không phải trần.
- **Metric (§6 + V2.6):**
  - AUC và PR-AUC, với nhãn cửa sổ và nhãn sự kiện;
  - recall@b và precision@b, b ∈ {0.1, 0.5, 1, 5}%, k = round(b·N), điểm hoà ở biên tính theo kỳ vọng; trần min(1, b·N/P);
  - tỉ lệ insider phát hiện @b;
  - số ngày tới cảnh báo đầu tiên với ngưỡng τ_b lấy từ phân vị điểm trên **val**;
  - F1 ở ngưỡng T2 (ngưỡng chọn trên val) là metric phụ.

---

## 6. Kết quả

Mỗi ô: trung bình ± std [min, max] qua 10 fold (PB-U), hoặc qua 5 seed (PB-T, chỉ với IF/HGB). Baseline tất định ở PB-T chỉ có một giá trị. Dòng in nghiêng là tham chiếu.

### 6.1 AUC (nhãn cửa sổ)
<!--TABLE:b1auc-->
| Baseline | r4.2 PBU | r4.2 PBT | r6.2 PBU | r6.2 PBT |
|---|---|---|---|---|
| Ngẫu nhiên (giải tích) | 0.500 | 0.500 | 0.500 | 0.500 |
| Khối lượng — chiều chọn trên train | 0.546 ± 0.087 [0.447, 0.700] | 0.408 | 0.768 ± 0.143 [0.517, 0.970] | 0.739 |
| Khối lượng +log(dòng) | 0.557 ± 0.080 [0.453, 0.700] | 0.592 | 0.768 ± 0.143 [0.517, 0.970] | 0.739 |
| Khối lượng −log(dòng) | 0.443 ± 0.080 [0.300, 0.547] | 0.408 | 0.232 ± 0.143 [0.030, 0.483] | 0.261 |
| Ô hiếm (tỉ lệ dòng) | 0.610 ± 0.034 [0.527, 0.656] | 0.592 | 0.674 ± 0.218 [0.369, 0.989] | 0.620 |
| Luật ngoài giờ + USB lần đầu ¹ | 0.588 ± 0.031 [0.534, 0.645] | 0.565 | 0.610 ± 0.138 [0.474, 0.816] | 0.514 |
| z-score: max \|z\| | 0.790 ± 0.012 [0.767, 0.815] | 0.797 | 0.866 ± 0.132 [0.661, 0.999] | 0.718 |
| z-score: tổng z dương | 0.811 ± 0.014 [0.787, 0.837] | 0.827 | 0.825 ± 0.174 [0.536, 1.000] | 0.634 |
| Isolation Forest G+D | 0.824 ± 0.023 [0.800, 0.867] | 0.839 ± 0.006 [0.833, 0.847] | 0.810 ± 0.151 [0.533, 0.980] | 0.750 ± 0.012 [0.734, 0.764] |
| Isolation Forest G+D+R | — | — | 0.822 ± 0.139 [0.563, 0.963] | 0.744 ± 0.006 [0.736, 0.751] |
| *Tham chiếu có giám sát:* HGB G+D | 0.880 ± 0.037 [0.798, 0.923] | 0.825 ± 0.067 [0.724, 0.927] | 0.392 ± 0.252 [0.014, 0.757] | 0.424 ± 0.120 [0.308, 0.592] |
| *Tham chiếu có giám sát:* HGB G+D+R | — | — | 0.378 ± 0.255 [0.014, 0.700] | 0.429 ± 0.081 [0.317, 0.534] |
| *Có thông tin kịch bản:* IF G+D+I | 0.829 ± 0.019 [0.806, 0.864] | 0.849 ± 0.008 [0.840, 0.862] | 0.827 ± 0.134 [0.590, 0.983] | 0.756 ± 0.013 [0.739, 0.774] |
| *Có thông tin kịch bản:* HGB G+D+I | 0.878 ± 0.055 [0.776, 0.967] | 0.866 ± 0.069 [0.734, 0.922] | 0.479 ± 0.293 [0.006, 0.861] | 0.386 ± 0.054 [0.308, 0.469] |
<!--/TABLE-->

### 6.2 PR-AUC (tỉ lệ dương: r4.2 ~0.4%, r6.2 0.01–0.02%)
<!--TABLE:b1pr-->
| Baseline | r4.2 PBU | r4.2 PBT | r6.2 PBU | r6.2 PBT |
|---|---|---|---|---|
| Ngẫu nhiên (giải tích) | 0.0041 ± 0.0011 [0.0020, 0.0054] | 0.0043 | 0.0001 ± 0.0001 [0.0000, 0.0002] | 0.0002 |
| Khối lượng — chiều chọn trên train | 0.0053 ± 0.0015 [0.0031, 0.0079] | 0.0034 | 0.0005 ± 0.0009 [0.0000, 0.0024] | 0.0010 |
| Khối lượng +log(dòng) | 0.0053 ± 0.0015 [0.0032, 0.0079] | 0.0055 | 0.0005 ± 0.0009 [0.0000, 0.0024] | 0.0010 |
| Khối lượng −log(dòng) | 0.0040 ± 0.0011 [0.0023, 0.0058] | 0.0034 | 0.0001 ± 0.0001 [0.0000, 0.0002] | 0.0001 |
| Ô hiếm (tỉ lệ dòng) | 0.0076 ± 0.0020 [0.0046, 0.0105] | 0.0063 | 0.0003 ± 0.0002 [0.0000, 0.0006] | 0.0003 |
| Luật ngoài giờ + USB lần đầu ¹ | 0.0120 ± 0.0075 [0.0051, 0.0317] | 0.0061 | 0.0002 ± 0.0001 [0.0000, 0.0004] | 0.0002 |
| z-score: max \|z\| | 0.0248 ± 0.0046 [0.0182, 0.0337] | 0.0269 | 0.0032 ± 0.0033 [0.0004, 0.0091] | 0.0013 |
| z-score: tổng z dương | 0.0418 ± 0.0100 [0.0283, 0.0639] | 0.0396 | 0.0477 ± 0.0877 [0.0004, 0.2599] | 0.0136 |
| Isolation Forest G+D | 0.0190 ± 0.0101 [0.0081, 0.0457] | 0.0135 ± 0.0006 [0.0127, 0.0144] | 0.0005 ± 0.0003 [0.0000, 0.0013] | 0.0007 ± 0.0000 [0.0007, 0.0008] |
| Isolation Forest G+D+R | — | — | 0.0004 ± 0.0003 [0.0000, 0.0008] | 0.0007 ± 0.0000 [0.0006, 0.0007] |
| *Tham chiếu có giám sát:* HGB G+D | 0.5857 ± 0.1082 [0.4325, 0.7350] | 0.7122 ± 0.0798 [0.5937, 0.8418] | 0.0002 ± 0.0002 [0.0000, 0.0007] | 0.0013 ± 0.0018 [0.0002, 0.0049] |
| *Tham chiếu có giám sát:* HGB G+D+R | — | — | 0.0001 ± 0.0002 [0.0000, 0.0004] | 0.0009 ± 0.0008 [0.0002, 0.0020] |
| *Có thông tin kịch bản:* IF G+D+I | 0.0211 ± 0.0115 [0.0086, 0.0533] | 0.0146 ± 0.0010 [0.0135, 0.0161] | 0.0007 ± 0.0007 [0.0000, 0.0027] | 0.0007 ± 0.0000 [0.0007, 0.0008] |
| *Có thông tin kịch bản:* HGB G+D+I | 0.6454 ± 0.0999 [0.4976, 0.7728] | 0.7401 ± 0.0697 [0.6101, 0.7927] | 0.0017 ± 0.0034 [0.0000, 0.0098] | 0.0025 ± 0.0019 [0.0003, 0.0049] |
<!--/TABLE-->

### 6.3 recall@1% user-day (%) — kèm trần
<!--TABLE:b1r1-->
| Baseline | r4.2 PBU | r4.2 PBT | r6.2 PBU | r6.2 PBT |
|---|---|---|---|---|
| **Trần recall @1%** | 100.0 | 100.0 | 100.0 | 100.0 |
| Ngẫu nhiên (giải tích) | 1.0 | 1.0 | 1.0 | 1.0 |
| Khối lượng — chiều chọn trên train | 0.5 ± 1.1 [0.0, 3.8] | 0.6 | 0.0 | 0.0 |
| Khối lượng +log(dòng) | 0.1 ± 0.2 [0.0, 0.6] | 0.0 | 0.0 | 0.0 |
| Khối lượng −log(dòng) | 1.5 ± 1.4 [0.0, 4.5] | 0.6 | 0.0 | 0.0 |
| Ô hiếm (tỉ lệ dòng) | 4.0 ± 2.3 [1.0, 9.6] | 2.2 | 13.3 ± 19.4 [0.0, 50.0] | 0.0 |
| Luật ngoài giờ + USB lần đầu ¹ | 3.3 ± 2.7 [1.3, 10.0] | 0.8 | 3.3 ± 6.7 [0.0, 16.7] | 0.0 |
| z-score: max \|z\| | 10.4 ± 2.1 [7.5, 15.6] | 10.8 | 52.3 ± 40.4 [8.9, 100.0] | 15.1 |
| z-score: tổng z dương | 15.3 ± 3.1 [10.1, 20.7] | 13.3 | 54.6 ± 39.8 [7.3, 100.0] | 12.8 |
| Isolation Forest G+D | 5.3 ± 3.4 [2.3, 13.9] | 2.3 ± 0.2 [2.2, 2.8] | 16.8 ± 19.3 [0.0, 50.0] | 4.7 |
| Isolation Forest G+D+R | — | — | 18.5 ± 19.9 [0.0, 50.0] | 3.5 |
| *Tham chiếu có giám sát:* HGB G+D | 83.6 ± 4.4 [72.6, 88.4] | 81.2 ± 7.1 [70.8, 92.5] | 10.2 ± 19.9 [0.0, 50.0] | 2.4 ± 1.9 [0.0, 5.4] |
| *Tham chiếu có giám sát:* HGB G+D+R | — | — | 0.9 ± 2.3 [0.0, 7.7] | 6.3 ± 5.8 [0.0, 15.2] |
| *Có thông tin kịch bản:* IF G+D+I | 5.5 ± 3.7 [1.7, 14.8] | 2.7 ± 0.3 [2.2, 3.1] | 16.8 ± 19.3 [0.0, 50.0] | 4.4 ± 0.5 [3.5, 4.7] |
| *Có thông tin kịch bản:* HGB G+D+I | 83.9 ± 5.6 [72.6, 93.9] | 84.2 ± 6.5 [71.9, 91.4] | 13.0 ± 26.6 [0.0, 78.0] | 6.3 ± 2.9 [3.5, 11.9] |
<!--/TABLE-->

### 6.4 Tỉ lệ insider test có ≥ 1 user-day dương lọt top 1% (%)
<!--TABLE:b1det-->
| Baseline | r4.2 PBU | r4.2 PBT | r6.2 PBU | r6.2 PBT |
|---|---|---|---|---|
| Ngẫu nhiên (giải tích) | — | — | — | — |
| Khối lượng — chiều chọn trên train | 4 ± 9 [0, 29] | 11 | 0 | 0 |
| Khối lượng +log(dòng) | 1 ± 3 [0, 7] | 0 | 0 | 0 |
| Khối lượng −log(dòng) | 17 ± 13 [0, 43] | 11 | 0 | 0 |
| Ô hiếm (tỉ lệ dòng) | 40 ± 13 [14, 57] | 28 | 40 ± 49 [0, 100] | 0 |
| Luật ngoài giờ + USB lần đầu ¹ | 27 ± 9 [14, 43] | 17 | 20 ± 40 [0, 100] | 0 |
| z-score: max \|z\| | 81 ± 9 [71, 100] | 72 | 100 | 100 |
| z-score: tổng z dương | 94 ± 7 [79, 100] | 78 | 100 | 100 |
| Isolation Forest G+D | 46 ± 11 [29, 64] | 29 ± 2 [28, 33] | 60 ± 49 [0, 100] | 50 |
| Isolation Forest G+D+R | — | — | 70 ± 46 [0, 100] | 50 |
| *Tham chiếu có giám sát:* HGB G+D | 95 ± 6 [79, 100] | 97 ± 3 [94, 100] | 25 ± 41 [0, 100] | 36 ± 21 [1, 58] |
| *Tham chiếu có giám sát:* HGB G+D+R | — | — | 15 ± 33 [0, 100] | 45 ± 33 [0, 100] |
| *Có thông tin kịch bản:* IF G+D+I | 44 ± 11 [29, 64] | 36 ± 6 [28, 44] | 60 ± 49 [0, 100] | 50 |
| *Có thông tin kịch bản:* HGB G+D+I | 99 ± 3 [93, 100] | 99 ± 2 [94, 100] | 20 ± 40 [0, 100] | 60 ± 20 [50, 99] |
<!--/TABLE-->

Các bảng ở mọi ngân sách (0.1/0.5/1/5%), precision@b, nhãn sự kiện, F1 T2, ngày tới cảnh báo, **theo kịch bản r4.2 (V2.9)**, **từng insider r6.2** và phân tích đơn biến đều nằm trong `results/phaseB_tables.md`.

### 6.5 Đọc kết quả

1. **So với giai đoạn A (câu hỏi 1).** P3 user-day ở giai đoạn A, r4.2:
   - AE R1: AUC 0.63, PR-AUC 0.008;
   - VAE R1: 0.80 / 0.040;
   - AE R2 và VAE R2: 0.83 / 0.021.

   Ở B1, baseline không giám sát tốt nhất có AUC 0.81–0.84 và PR-AUC 0.019–0.042. Như vậy **ngang**, chưa "vượt xa". F1 T2 tốt nhất: z tổng dương 8.3% so với ≤ 5% ở giai đoạn A.
2. **Đặc trưng chứa thông tin, nhưng điểm bất thường không giám sát không khai thác được.**
   - HGB trên cùng G+D đạt PR-AUC 0.59 (PB-U) và 0.71 (PB-T); recall@1% 84%.
   - Theo kịch bản (V2.9, PB-U), HGB mạnh nhất ở kịch bản 2 (PR-AUC 0.59, R@1% 86%). Baseline không giám sát lại yếu ở kịch bản này: z tổng dương 0.026, IF 0.014.
   - Phân tích đơn biến trên tập fit (sau khi xem kết quả): `z_g8_http` một mình đạt PR-AUC 0.51–0.55 dù AUC chỉ 0.56–0.60. Nghĩa là một phần nhỏ ngày dương có đột biến lượt http cực lớn so với lịch sử của chính user, và ở đầu bảng xếp hạng các ngày này gần như toàn là dương.
   - **Giới hạn của quan sát này:** đột biến có thể một phần là dấu vết của cách CERT chèn hoạt động độc hại vào dữ liệu tổng hợp (khối lượng http của ngày kịch bản được sinh khác với nền), chứ không nhất thiết là hành vi của insider thật. Chưa suy rộng cho dữ liệu thật.
   - Tổng z dương cộng 14 chiều nên làm loãng tín hiệu này. Cách xếp hạng không giám sát nào bắt được nó **mà không cần chọn đặc trưng bằng nhãn** là câu hỏi của B2. Chúng tôi **không** dùng quan sát này để tạo baseline mới, vì làm vậy là chọn theo nhãn.
3. **Theo kịch bản (r4.2, PB-U).**
   - Kịch bản 1 (USB ngoài giờ, wikileaks): ô hiếm và luật có AUC 0.73–0.77; z tổng dương có R@1% 37%.
   - Kịch bản 3 (10 insider, 2–8 dương mỗi fold): IF và z-score có AUC 0.96–0.98.
   - Kịch bản 2 chiếm ~87% dương test nên quyết định PR-AUC chung.
4. **Mức insider.** Mức insider dễ hơn mức user-day rất nhiều:
   - z tổng dương phát hiện 94% insider (PB-U) và 78% (PB-T) ở ngân sách 1%; recall user-day chỉ 13–15%.
   - Trung vị số ngày tới cảnh báo (τ từ val) thường bằng 0: cảnh báo rơi ngay ngày đầu cửa sổ, vì ngày đó thường đã có hành vi kịch bản.
5. **Chia theo thời gian (câu hỏi 3).**
   - r4.2: PB-T gần PB-U ở mọi baseline. Nhưng test PB-T có 333/360 ngày dương (92%) thuộc kịch bản 2, kịch bản 1 chỉ 25 và kịch bản 3 chỉ 2. Vì vậy câu "gần PB-U" thực chất là kết luận về **kịch bản 2**; với kịch bản 1 và 3, PB-T không đủ dương để kết luận.
   - Khối lượng: dấu chọn trên train PB-T là "−" (AUC 0.41 trên test), trong khi PB-U chọn "+" ở 9/10 fold. Có lẽ vì insider rời sớm nên ở giai đoạn đầu họ hoạt động "ít hơn" (nhận xét sau khi xem kết quả).
   - r6.2: train PB-T chỉ có 9 dương, thuộc kịch bản khác test, nên HGB thất bại (AUC 0.42). z-score không cần nhãn nên vẫn đạt PR-AUC 0.014, và cả 2 insider đều lọt top 0.5%.
6. **r6.2 từng insider (PB-U)** (bảng trong `results/phaseB_tables.md`):
   - z max |z| đưa ngày dương tốt nhất của 4/5 insider vào top 0.2%, và của MBG3183 (kịch bản 5, Dropbox) vào top 0.8–0.9%.
   - HGB đẩy MBG3183 xuống ~98% (tức gần cuối bảng xếp hạng).
   - IF: 4/5 insider vào top 2.2%, riêng kịch bản 5 ở top 10–16%.
   - Ở r6.2, nhóm R (thao tác file, thiết bị rời) **không** cải thiện IF: PR-AUC 0.0004 so với 0.0005.
7. **Nhóm I ("có thông tin kịch bản")** chỉ tăng nhẹ: IF 0.019 → 0.021, HGB 0.59 → 0.65 (PB-U r4.2). Từ khoá không giải thích được khoảng cách giữa HGB và baseline không giám sát.

### 6.6 Trần

Với đặc trưng liên tục nhiều chiều, không có trần Bayes chặt (§7 spec). Thứ duy nhất có dạng trần là trần recall min(1, b·N/P), in kèm mọi bảng recall@b. Từ 0.5% trở lên, trần gần 100% (r4.2 PB-U @0.1%: 26.5%). HGB là **tham chiếu có giám sát**, không phải trần.

---

## 7. Lỗi, cảnh báo và chỗ chưa chắc

**Lỗi đã gặp (theo thứ tự):**
1. **Hết RAM khi trích r6.2 (lần 1).**
   - Ở bước `http` (90 GB, 117 triệu dòng), polars báo `memory allocation of 16777216 bytes failed`.
   - Lúc đó RAM cạn khiến một lệnh đọc kết quả chạy song song cũng lỗi (OpenBLAS không cấp được bộ nhớ) và bash không fork được.
   - Mình đã chuyển các biểu thức có điều kiện ra khỏi `agg`; r4.2 cho kết quả giống hệt.
2. **Hết RAM (lần 2).** Working set lên 21 GB và RAM trống còn 0.1 GB, nên mình đã dừng tiến trình của chính mình.
   - Đo lại cho thấy chỉ cần dùng cột url theo bất kỳ cách nào (kể cả `str.len_bytes()`) là trình đọc CSV streaming của polars 2.0 vượt 9 GB sau khoảng 11 giây. Không dùng url thì chạy bình thường (~4 phút). Kết luận: trình đọc giữ cột url mà không có giới hạn.
   - **Sửa:** riêng file http đọc bằng `pyarrow.csv.open_csv` theo khối và gộp từng phần bằng polars (`userday.http_tables`).
   - Khối 256 MB vẫn vượt 20 GB vì pyarrow đọc trước nhiều khối song song (RSS ≈ 70 × kích thước khối), nên bị bộ giám sát dừng. Khối 32 MB thì ổn định quanh 2.4 GB.
   - **Kiểm chứng:** với r4.2, cả bảng `http` lẫn `http_dom` theo đường mới (khối 256 MB và 32 MB) giống hệt bản polars streaming. r6.2 đọc đủ 117 025 216 dòng, đúng bằng số dòng của giai đoạn A.
   - Việc tách theo khối phụ thuộc điều kiện không có giá trị nào kéo dài qua nhiều dòng (rows = lines − 1 ở mọi file, đã đo ở giai đoạn A).
3. **Lỗi sklearn.** `HistGradientBoostingClassifier` không có `_use_early_stopping` mà dùng `do_early_stopping_`. Lần chạy r4.2 PB-T đầu tiên vì vậy bị dừng **sau khi** điểm của các baseline tất định đã được chấm trên test, nhưng chưa in hay lưu.
   - Chạy lại đúng cấu hình đó, không đổi gì; điểm tất định nên giống hệt.
   - Ghi lại vì quy tắc "test chỉ đọc một lần cho mỗi cấu hình".
4. **Lỗi trình bày bảng:** dấu `|` trong "max |z|" làm vỡ bảng markdown; bảng đơn biến bị dính nhau; tỉ lệ dương r6.2 bị làm tròn về 0. Đã sửa, không đổi số.
5. Bộ giám sát RAM (`memguard`, script tạm) lần đầu không tìm được python qua đường dẫn tương đối. Chạy lại với đường dẫn tuyệt đối.

**Chỗ phải tự quyết / chưa chắc** (đều chốt trong spec trước khi chạy):
- G3 khi thiếu logon/logoff thì lấy dòng đầu/cuối ngày.
- Nhóm D tính trên số đếm **thô**, σ dùng ddof = 0.
- G4 = 0 khi chưa có logon trước d (V2.2 không định nghĩa "chưa có lịch sử" cho G4 bằng 10 ngày; chúng tôi hiểu là chưa có logon nào).
- "Vượt τ_b" là so sánh chặt; điểm hoà ở biên top-k tính theo kỳ vọng.
- Phần mở rộng từ khoá nhóm I (career, resume, …) không có nguyên văn trong `scenarios.txt`.
- Chuỗi con "job" khớp cả URL ngẫu nhiên, nên nhóm I2 nhiễu.

**Cảnh báo khi đọc số:**
- **Tỉ lệ dương rất nhỏ:**
  - r6.2 PB-U: mỗi fold test có 1 insider với 1–45 ngày dương;
  - r6.2 PB-T: 2 insider, 86 ngày dương.
  - Vì vậy std rất lớn và kết luận cho r6.2 rất yếu.
- **HGB dừng sớm ở 10–11 vòng** (mặc định). Không tinh chỉnh nên con số có thể thấp hơn mức đạt được. Ở r6.2, val PB-U chỉ có 1 insider, còn PB-T không có dương, nên F1 T2 ở r6.2 là "—" hoặc ~0.
- **Phân tích đơn biến và bảng V2.10 là phân tích sau**, chỉ dùng tập fit (đơn biến) hoặc mô tả (V2.10), không dùng để chọn gì. Riêng tín hiệu `z_g8_http`: có thể một phần là dấu vết của cách CERT chèn hoạt động độc hại vào dữ liệu tổng hợp; chưa suy rộng cho dữ liệu thật.
- **Test (e) là kiểm tra thiết kế, không phải kiểm tra thực nghiệm.** Test xác nhận hai điều:
  - chữ ký của `userday.extract` và `userday.assemble` không có tham số nào về nhãn hay cách chia;
  - chạy lại bộ trích trên cùng dữ liệu tổng hợp cho bảng giống hệt, tức là tất định; chỉ bộ chuẩn hoá thay đổi theo cách chia.

  Test **không** chạy bộ trích trên dữ liệu thật với hai danh sách insider khác nhau rồi so sánh, như câu chữ của V2.8. Lý do: hàm không nhận danh sách đó, nên không có gì để thay đổi. Kết luận "đặc trưng không phụ thuộc cách chia" vì vậy dựa trên thiết kế mã, không dựa trên phép đo.
- **Ngày tới cảnh báo bằng 0** không phải rò rỉ: τ lấy từ val, đặc trưng chỉ dùng quá khứ (test a), và ngày đầu cửa sổ thường đã có hành vi bất thường.
- **Quy tắc V2.12** có thể hiểu "trung bình tăng tương đối ≥ 10%" theo hai cách: (PR-AUC trung bình của mô hình / PR-AUC trung bình của baseline − 1), hoặc trung bình của tỉ số từng fold. **Đã chốt ở mục B2-0 của spec (2026-10-17):** tiêu chí chính là tỉ số các trung bình; báo kèm trung bình của tỉ số từng fold; nếu hai cách cho kết luận khác nhau thì ghi "không kết luận".

---

## 8. Kết luận B1

- **Câu hỏi 1:** đặc trưng user-day chứa đủ thông tin để một mô hình **có giám sát** đạt PR-AUC 0.59–0.71 ở r4.2. Tuy vậy, các baseline **không giám sát** chỉ đạt mức ngang giai đoạn A ở mức user-day: PR-AUC ≤ 0.042, recall@1% ≤ 15%. Ở mức insider, z-score với ngân sách 1% đã bắt được 78–94% insider r4.2.
- **Câu hỏi 3:** ở r4.2, kết quả giữ được khi chia theo thời gian, **nhưng chỉ cho kịch bản 2**: test PB-T có 333/360 ngày dương (92%) thuộc kịch bản 2, kịch bản 1 chỉ 25 và kịch bản 3 chỉ 2. r6.2 quá ít insider để kết luận; chỉ z-score còn tín hiệu.
- **Câu hỏi 2** để cho B2. Mốc đã khoá: **z-score tổng z dương, PB-U r4.2, PR-AUC 0.0418**. Muốn được ghi "hơn baseline", mô hình phải cao hơn ở ≥ 8/10 fold và tỉ số các PR-AUC trung bình ≥ 1.10 (tức ≥ 0.0460). Deep SAD (dùng nhãn) còn được so với HGB G+D. Quy tắc đầy đủ ở mục B2-0 của spec.
- **B2** (chưa làm, chờ duyệt): Deep SVDD, Deep SAD và Temporal AE. Chi tiết chạy (chuẩn hoá, kiến trúc, lưới, seed, ước lượng thời gian) chốt ở mục **"B2-0 — 2026-10-17"** của `docs/phaseB_spec.md`.

---

## 9. Cách chạy lại

```bash
# đặc trưng (r4.2 ~1 phút; r6.2 ~8.5 phút, RAM đỉnh ~13 GB)
python src/userday.py 4.2
python src/userday.py 6.2
# test (a)–(e) + test cũ
pytest tests
# baseline B1 (chạy từ thư mục src/)
cd src
python baselines_b.py 4.2 PBT && python baselines_b.py 4.2 PBU
python baselines_b.py 6.2 PBT && python baselines_b.py 6.2 PBU
# bảng + điền bảng vào REPORT_B.md
python tables_b.py
```

Đầu ra:
- `data/userday/` (không commit);
- `results/phaseB/*.json` (từng lần chạy, tham số, phiên bản);
- `results/phaseB/feature_schema_r*.json` (hash schema);
- `results/phaseB/describe.json`;
- `results/phaseB_tables.md`;
- log ở `logs/phaseB_*.log`.
