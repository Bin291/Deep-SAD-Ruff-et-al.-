# Giai đoạn B — Đặc tả và pre-registration (B0)

Viết ngày 2026-10-16, **trước** khi viết code đặc trưng hay mô hình nào của giai đoạn B và trước khi xem bất kỳ kết quả nào trên các đặc trưng này. Mọi lựa chọn dưới đây cố định. Nếu phải thay đổi thì ghi lại kèm lý do và ngày, và không được dựa trên kết quả test. Chỗ đánh dấu **[tự chọn]** là lựa chọn của chúng tôi.

Cổng: **B0 (tài liệu này) → người dùng duyệt → B1 (baseline) → báo cáo → người dùng duyệt → B2 (Deep SVDD, Deep SAD, Temporal AE).** Không làm psychometric/LDAP ở giai đoạn này. Không cài gói mới (đã đủ polars, numpy, pandas, scikit-learn, torch); nếu cần gói khác sẽ hỏi trước.

Liên quan: giai đoạn A — `REPORT.md`, `docs/step5_preregistration.md`.

---

## 1. Câu hỏi nghiên cứu

1. Khi thay đặc trưng của bài (activity, thứ, giờ của từng dòng) bằng **đặc trưng user-day**, phát hiện insider ở mức **user-day** có vượt xa giai đoạn A không? Đặc trưng user-day gồm khối lượng theo khung giờ, ngoài giờ, lệch so với lịch sử của chính user, domain http, email ra ngoài và đính kèm. Mốc so sánh là P3 r4.2 ở giai đoạn A: AUC ≤ 0.83, F1 ≤ ~5%.
2. Trên **cùng** đặc trưng, giao thức và ngân sách cảnh báo, mô hình sâu (B2) có hơn baseline đơn giản (B1) không?
3. Kết quả có giữ được khi chia theo **thời gian** (train giai đoạn đầu, test giai đoạn sau), không chỉ khi chia theo user?

---

## 2. Đơn vị, nhãn, dữ liệu

- **Đơn vị chính:** user-day = (user, ngày lịch) có ít nhất 1 dòng log. Ngày không có dòng nào thì không chấm.
- **Nhãn chính:** user-day dương nếu có ≥ 1 dòng của insider nằm trong **cửa sổ hợp** (như giai đoạn A, bộ insider `main`). r4.2 có 1 364 user-day dương trên 330 452; r6.2 có 95 trên 1 394 010.
- **Nhãn phụ:** (i) mức sự kiện: user-day có ≥ 1 sự kiện trong file chi tiết (r4.2: 966; r6.2: 44); (ii) mức user: là insider hay không.
- **Bản dữ liệu:** r4.2 và r6.2, bản local (sha256 khớp HF). Không dùng `psychometric.csv`, LDAP, `users.csv` (kể cả ngày nghỉ việc). PC "của user" được suy ra từ log, không lấy từ LDAP.

---

## 3. Đặc trưng user-day

Tính bằng **một lượt stream** qua 5 file log của mỗi bản, gom theo (user, ngày). Phần lịch sử được tính sau, trên bảng user-day đã sắp theo (user, ngày). Các giá trị đếm được biến đổi bằng `log1p`.
- **Khung giờ:** K0 = 00–06, K1 = 06–12, K2 = 12–18, K3 = 18–24 **[tự chọn]**.
- **Ngoài giờ:** ngày thường trước 08:00 hoặc từ 18:00, cộng mọi giờ cuối tuần **[tự chọn]**.

### 3.1 Nhóm G — chung cho cả hai bản (đặc trưng chính)

| # | Nhóm | Đặc trưng | Số chiều |
|---|---|---|---|
| G1 | Khối lượng × khung giờ | đếm {logon, logoff, connect, disconnect, file, email, http} × {K0..K3} | 28 |
| G2 | Ngoài giờ | đếm ngoài giờ của {logon, connect, file, email, http}; cờ cuối tuần | 6 |
| G3 | Phiên làm việc | giờ logon đầu, giờ logoff cuối (thập phân), độ dài (cuối − đầu) | 3 |
| G4 | PC | số PC khác nhau; số logon trên PC **không phải** PC chính của user (PC chính = PC có nhiều logon nhất trong **quá khứ** của user, trước ngày d) | 2 |
| G5 | USB | số connect; cờ "lần dùng USB đầu tiên" (không có connect nào trước ngày d) | 2 |
| G6 | File | số dòng file; đếm theo nhóm đuôi {doc/pdf/txt, exe/dll, zip/rar, jpg/png, khác} | 6 |
| G7 | Email | số email gửi; số người nhận ngoài (địa chỉ không kết thúc `@dtaa.com`) trong to/cc/bcc; số người nhận bcc ngoài; số email có đính kèm; tổng số đính kèm; kích thước trung bình; số người nhận khác nhau | 7 |
| G8 | HTTP chung | số lượt truy cập; số domain khác nhau; số domain **mới với user** (chưa từng truy cập trước ngày d); số domain **mới với toàn tổ chức** (chưa ai truy cập trước ngày d) | 4 |

Domain = phần host của URL (`http://host/...`), viết thường.

### 3.2 Nhóm R — chỉ có ở r6.2 (nhờ các trường bổ sung)

r6.2 có thêm các trường sau:
- file: thao tác {open, write, copy, delete}, `to_removable_media`, `from_removable_media`;
- email: {Send, View}; G7 chỉ đếm Send, còn View được đếm riêng;
- http: {Visit, Download, Upload};
- danh sách file mồi `decoy_file.csv`.

Đặc trưng thêm (**≈ 10 chiều**): đếm 4 loại thao tác file, đếm ghi ra / đọc từ thiết bị rời, số email View, số Download, số Upload, số lần chạm file mồi.

Kết quả r6.2 được báo với cả G và G+R.

### 3.3 Nhóm D — lệch so với lịch sử của chính user (chỉ dùng quá khứ)

Áp dụng cho tập con S (≈ 14 đặc trưng): connect, file, email gửi, người nhận ngoài, email có đính kèm, http, domain mới với user, logon trên PC khác, mỗi đếm ngoài giờ ở G2, độ dài phiên. Với mỗi đặc trưng trong S, tính:
- z_d = (x_d − μ_{<d}) / (σ_{<d} + 1), trong đó μ, σ tính trên **các user-day trước ngày d của chính user** (cửa sổ mở rộng) **[tự chọn: +1 ở mẫu số để ổn định]**;
- tỉ lệ x_d / (trung bình của 30 ngày hoạt động trước đó + 1);
- nếu user có < 10 ngày hoạt động trước d: đặt z = 0, tỉ lệ = 1, và cờ `lich_su_ngan` = 1.

Nhóm D có **≈ 29 chiều**. Tổng G + D ≈ 87 chiều; r6.2 thêm R thành ≈ 97.

### 3.4 Nhóm I — "biết trước kịch bản" (chỉ để tham chiếu lạc quan, **không** thuộc đặc trưng chính)

Nhóm này dùng kiến thức về các kịch bản red-team: từ khoá lấy từ `answers/scenarios.txt`. Đặc trưng là số lượt http có domain/URL chứa một trong các nhóm từ khoá:
- {`wikileaks`};
- {`job`, `career`, `resume`, `hiring`, `recruit`};
- {`dropbox`, `upload`, `storage`, `drive`};
- {`keylog`, `spy`}.

Danh sách cố định từ bây giờ. Mọi kết quả dùng nhóm I phải ghi rõ **"có thông tin kịch bản"** và không được so ngang với kết quả chỉ dùng G/R/D.

---

## 4. Quy tắc chống rò rỉ

1. **Chỉ dùng quá khứ:** đặc trưng của ngày d chỉ dùng dữ liệu của các ngày < d (lịch sử) và của chính ngày d. Quy tắc này áp dụng cho PC chính (G4), lần USB đầu (G5), domain mới (G8) và toàn bộ nhóm D.
2. **Chuẩn hoá** (mean/std, phân vị) chỉ ước lượng trên user-day của **tập train** (user train ở PB-U; giai đoạn train ở PB-T), rồi áp nguyên cho val/test.
3. **Thống kê toàn tổ chức** (domain mới với tổ chức) được tích luỹ theo thời gian, chỉ dùng quá khứ, và không phụ thuộc cách chia user.
4. **Không dùng thông tin suy ra từ nhãn** trong đặc trưng: ngày bắt đầu/kết thúc kịch bản, ngày nghỉ việc, danh sách insider.
5. **Ngưỡng và lựa chọn mô hình** chỉ dùng train/val. Với mỗi cấu hình đã chốt, tập test chỉ được đọc một lần.
6. **Kiểm thử bắt buộc ở B1:**
   - (a) đổi ngẫu nhiên dữ liệu của mọi ngày > d thì đặc trưng của các ngày ≤ d không đổi;
   - (b) tập user / khoảng ngày của train, val, test rời nhau;
   - (c) tổng đếm G1 khớp bảng đếm của giai đoạn A;
   - (d) parser domain/email chạy đúng trên ví dụ tổng hợp.

---

## 5. Giao thức chia

| Mã | Chia | Train | Val | Test | Số lần chạy |
|---|---|---|---|---|---|
| **PB-U** | theo user — **dùng đúng các fold P3 của giai đoạn A** (r4.2: 5-fold phân tầng × 2 lần lặp; r6.2: leave-one-insider-out × 2) | user-day của user fit | user-day của user val | user-day của user test | 10 |
| **PB-T** | theo thời gian, mọi user | cả hai bản: 2010-01-02 → 2010-10-31 | cả hai bản: 2010-11-01 → 2010-12-31 | r4.2: 2011-01-01 → 2011-05-17; r6.2: 2011-01-01 → 2011-06-01 | 1 lần chia × 5 seed (mô hình ngẫu nhiên) |

- **Cách chọn mốc PB-T:** chỉ dựa trên phân bố nhãn theo tháng, không nhìn kết quả nào.
  - r4.2: train có 778 user-day dương, val 226, test 360. Test có ≥ 13 insider khác nhau; số chính xác sẽ báo ở B1.
  - r6.2: train có 9 user-day dương (3 insider, kịch bản 1, 3, 5), val có 0, test có 86 (2 insider, kịch bản 2, 4).
- Ở PB-T, insider nào có cửa sổ vắt qua mốc chia sẽ góp user-day dương vào cả hai phía (mỗi user-day tính theo ngày của nó). Đây là đặc điểm của chia theo thời gian và sẽ được ghi rõ.
- Đặc trưng lịch sử của ngày test được phép dùng dữ liệu giai đoạn trước, vì đó là quá khứ.
- Val PB-T của r6.2 không có dương. Vì vậy ở đó không dùng ngưỡng theo nhãn; mô hình không giám sát chỉ dùng val để dừng sớm theo loss.

---

## 6. Metric

Metric chính không phụ thuộc ngưỡng, vì cách dùng thực tế là theo ngân sách cảnh báo.
- **AUC-ROC và PR-AUC** ở mức user-day (nhãn cửa sổ); báo thêm với nhãn sự kiện.
- **Ngân sách cảnh báo:** xếp hạng mọi user-day của tập test, gắn cờ top b% với b ∈ {0.1%, 0.5%, 1%, 5%} **[tự chọn]**. Với toàn bộ r4.2 (~660 user-day/ngày), b = 0.1% tương đương khoảng 0.7 cảnh báo/ngày. Báo **recall@b** và **precision@b** ở mức user-day.
- **Mức insider:**
  - tỉ lệ insider trong test có ≥ 1 user-day dương lọt top b%;
  - trung vị số ngày từ đầu cửa sổ tới cảnh báo đầu tiên.
- **r6.2:** không lấy trung bình F1. Báo **từng insider** (5 người) ở cả PB-U và PB-T:
  - thứ hạng phân vị của user-day dương tốt nhất;
  - có lọt top b% không;
  - số ngày tới cảnh báo đầu tiên.
- **Cách trình bày:** mỗi số ghi trung bình ± std [min, max] qua các lần chạy, kèm số user-day dương và số insider của mỗi tập test.
- **F1 ở ngưỡng T2** (như giai đoạn A) chỉ là metric phụ, để nối với báo cáo giai đoạn A.

---

## 7. Baseline (B1) và tham chiếu

**B1 — baseline (phải chạy trước mô hình sâu):**
1. **Ngẫu nhiên:** AUC 0.5; recall@b = b.
2. **Khối lượng:** log tổng số dòng trong user-day.
3. **Ô hiếm** từ giai đoạn A: tỉ lệ dòng ô hiếm trong user-day, với ô hiếm tính trên train. Baseline này là cầu nối từ giai đoạn A sang B.
4. **Luật:** số đếm ngoài giờ của connect + file (G2), cộng cờ lần dùng USB đầu.
5. **z-score theo user:** max |z| và tổng z dương trên nhóm D.
6. **Isolation Forest** **[tự chọn]**:
   - scikit-learn; 200 cây, max_samples 256, các tham số khác để mặc định;
   - đặc trưng G + D; r6.2 thêm biến thể G + D + R;
   - học trên user-day train, không dùng nhãn (chấp nhận lẫn ~0.4% dương).

**Tham chiếu (không phải trần):**
- **Có giám sát:** HistGradientBoosting (scikit-learn, tham số mặc định) trên G + D, dùng nhãn train. Cho biết đặc trưng chứa bao nhiêu thông tin khi có nhãn.
- **Có thông tin kịch bản:** Isolation Forest và mô hình có giám sát trên G + D + I.
- **Giai đoạn A:** các số P3 cũ (AUC user-day ≤ 0.83 ở r4.2).

**Về trần:** với đặc trưng liên tục nhiều chiều, không tính được trần Bayes chính xác như ở giai đoạn A. Thay vào đó báo tham chiếu có giám sát ở trên. Mức "oracle" tầm thường (AUC 1) không có ý nghĩa nên không báo. Báo cáo B sẽ nói rõ là không có trần chặt.

---

## 8. Mô hình B2 (chỉ đặc tả; code sau khi báo B1 và được duyệt)

**Dùng chung:** cùng đặc trưng (G + D; r6.2 thêm G + D + R), cùng PB-U/PB-T, cùng metric và baseline.

**Chọn siêu tham số:** lưới nhỏ được khai báo trước. Lựa chọn làm **trên val** theo tiêu chí không dùng test:
- mô hình không giám sát: loss trên val;
- Deep SAD: PR-AUC trên val, chỉ ở PB-U/r4.2 vì đó là nơi val có dương.

**Các mô hình:**
- **Deep SVDD** (one-class, Ruff 2018):
  - MLP d → 64 → 32, không bias; tâm c cố định bằng trung bình sau pretrain AE;
  - weight decay 1e-6; Adam lr 1e-4; 150 epoch; batch 256;
  - lưới: chiều ra {16, 32}.
- **Deep SAD** (Ruff 2020):
  - cùng mạng; nhãn dương lấy từ user-day train (cửa sổ); η = 1;
  - lưới: η ∈ {0.5, 1, 2};
  - không dùng ở r6.2 PB-T (train chỉ có 9 dương), trừ biến thể được ghi rõ.
- **Temporal AE:**
  - chuỗi W user-day liên tiếp của một user → GRU encoder (64) → GRU decoder → tái tạo;
  - điểm của ngày d = lỗi tái tạo ngày cuối của cửa sổ kết thúc tại d (chỉ dùng quá khứ);
  - W ∈ {7, 14}; Adam lr 1e-3; 50 epoch.

**Số lần chạy:** mỗi cấu hình 10 lần (PB-U) hoặc 5 seed (PB-T); chạy trên GPU; cùng quy tắc tái lập (seed, deterministic) như giai đoạn A.

---

## 9. Kế hoạch tính toán

- **Trích đặc trưng:** 1 lượt stream mỗi file (polars), sinh bảng (user, ngày) → parquet `data/userday/r<rel>.parquet`. Chạy nền, log vào `logs/`. Ước tính:
  - r4.2: ~5 phút;
  - r6.2: ~20–40 phút; phần nặng nhất là tách domain của 117 triệu dòng http.
- **B1:** vài phút mỗi giao thức (Isolation Forest trên ~330 nghìn / 1.39 triệu user-day).
- **Đầu ra:**
  - kết quả từng lần chạy: `results/phaseB/…json`;
  - bảng: `results/phaseB_tables.md`;
  - báo cáo: `REPORT_B.md`, theo cùng dàn bài §5 của brief v2.

---

## 10. Những điểm cần người dùng duyệt

1. Danh sách đặc trưng G / R / D, và việc tách nhóm I ("có thông tin kịch bản") thành tham chiếu riêng.
2. Định nghĩa ngoài giờ (trước 08:00 / từ 18:00 ngày thường, cộng cuối tuần) và 4 khung giờ.
3. Mốc PB-T: train ≤ 2010-10-31, val 11–12/2010, test 2011, áp dụng cho cả hai bản.
4. Ngân sách cảnh báo {0.1%, 0.5%, 1%, 5%} user-day.
5. Isolation Forest và baseline có giám sát (HistGradientBoosting) chỉ dùng scikit-learn — không cài gì thêm.

---

## B0 v2 — 2026-10-16, trước khi có kết quả giai đoạn B

Người dùng duyệt B0 (5/5 điểm ở §10) kèm 12 bổ sung dưới đây. Mọi mục ghi ở đây (kể cả phần "chi tiết thực thi" cuối mục) được chốt **trước** khi trích đặc trưng và trước khi xem bất kỳ kết quả nào của giai đoạn B. Khi mâu thuẫn với phần trên, phần này thắng.

**V2.1 — Cold start.** Trong 30 ngày lịch đầu của mỗi bản (r4.2 và r6.2 đều bắt đầu 2010-01-02, nên là 2010-01-02 → 2010-01-31), hai đặc trưng "domain mới với user" và "domain mới với tổ chức" (G8) đặt = 0, và thêm cờ `khoi_dau` = 1 (ngoài khoảng đó = 0). Báo cáo B1 ghi số user-day dương nằm trong 30 ngày này.

**V2.2 — Chưa có lịch sử.**
- G4 "số logon trên PC không phải PC chính" = 0 khi user chưa có logon nào trước ngày d (chưa xác định được PC chính).
- Cờ "USB lần đầu" (G5) = 1 chỉ khi: ngày d có connect, chưa có connect nào trước d, **và** user đã có ≥ 10 ngày hoạt động (user-day có ít nhất 1 dòng log) trước d. Ngược lại = 0.

**V2.3 — Domain mới.** "Mới" nghĩa là chưa xuất hiện ở bất kỳ ngày nào < d (với user: trong log của chính user; với tổ chức: trong log của mọi user). Nếu nhiều user cùng truy cập một domain lần đầu trong cùng ngày d thì với mỗi user đó, domain đều tính là mới với tổ chức. Đơn vị đếm là số domain khác nhau (không phải số lượt).

**V2.4 — Nhóm I đối chiếu với scenarios.txt của từng bản.** Gói `answers` chỉ có **một** `scenarios.txt` dùng chung cho mọi bản; tập kịch bản của từng bản lấy từ `insiders.csv`: r4.2 = {1, 2, 3}, r6.2 = {1, 2, 3, 4, 5}. Mỗi nhóm từ khoá gắn với kịch bản sinh ra nó và chỉ được dùng ở bản có kịch bản đó:

| Nhóm | Kịch bản | Từ có nguyên văn trong scenarios.txt | Từ do chúng tôi mở rộng (không có nguyên văn) | r4.2 | r6.2 |
|---|---|---|---|---|---|
| I1 `wikileaks` | 1 | wikileaks | — | dùng | dùng |
| I2 `job` | 2 | job | career, resume, hiring, recruit | dùng | dùng |
| I3 `keylog` | 3 | keylog (trong "keylogger") | spy | dùng | dùng |
| I4 `dropbox` | 5 | dropbox; upload (kịch bản 1 và 5); drive (chỉ là "removable/thumb drive", tức thiết bị chứ không phải web) | storage | **không** (r4.2 không có kịch bản 5) | dùng |

Kịch bản 4 (gửi email về nhà) không có từ khoá web nên không có nhóm. Danh sách từ giữ nguyên như B0, không thêm/bớt. B1 sẽ báo thêm (chỉ mô tả) số lượt http khớp mỗi nhóm ở mỗi bản. Mọi kết quả dùng nhóm I chỉ được báo là tham chiếu **"có thông tin kịch bản"**.

**V2.5 — Bảng mô tả PB-T bắt buộc.** B1 in bảng kịch bản × giai đoạn (train/val/test) cho PB-T: số insider có ≥ 1 user-day dương trong giai đoạn đó và số user-day dương. r4.2 có kịch bản 1/2/3; r6.2 in cả 5. Bảng chỉ để mô tả; **không đổi mốc** sau khi thấy bảng.

**V2.6 — Trần recall và ngày tới cảnh báo.**
- Mọi bảng recall@b in kèm trần min(1, b·N/P), với N = số user-day test, P = số user-day dương test.
- "Số ngày tới cảnh báo đầu tiên" dùng **ngưỡng cố định** τ_b = phân vị (1 − b) của điểm trên user-day **val** (PB-U: user val của fold; PB-T: giai đoạn val). Không dùng phân vị gộp của test.

**V2.7 — Baseline khối lượng và luật.**
- Khối lượng: điểm = ± log(tổng số dòng của user-day). Chiều (dấu) được chọn bằng AUC trên user-day **train** (PB-U: user fit; PB-T: giai đoạn train); báo cả hai chiều, chiều đã chọn được đánh dấu.
- Luật: điểm = log1p(số connect ngoài giờ) + log1p(số dòng file ngoài giờ) + cờ USB lần đầu; ba thành phần trọng số bằng nhau, cố định, không tinh chỉnh. **Luật này được thiết kế khi đã biết hình dạng kịch bản** (kịch bản 1: ngoài giờ + USB), nên được báo như baseline "có hiểu biết miền", không phải baseline mù.

**V2.8 — Test (e).** Đặc trưng thô không phụ thuộc cách chia: chạy bộ trích đặc trưng với hai danh sách insider/cách chia khác nhau thì bảng đặc trưng phải giống hệt nhau. Chỉ bộ chuẩn hoá (fit trên train) phụ thuộc cách chia.

**V2.9 — Báo theo kịch bản (r4.2).** Mọi baseline báo AUC, PR-AUC, recall@b riêng cho kịch bản 1, 2, 3. Nhãn kịch bản chỉ dùng để báo cáo, không là đặc trưng. Định nghĩa: tập đánh giá kịch bản s = mọi user-day âm của test + user-day dương của insider kịch bản s (bỏ user-day dương của các kịch bản khác); recall@b kịch bản s = tỉ lệ user-day dương kịch bản s lọt top b% của **toàn bộ** test (ngân sách tính trên toàn bộ test).

**V2.10 — Bảng chẩn đoán "insider rời sớm".** Số ngày hoạt động và số dòng/ngày hoạt động, insider so với người thường, ở r4.2 và r6.2 (trung bình, trung vị). Kèm ngày hoạt động cuối cùng. Dùng để kiểm chứng giả thuyết ở REPORT §6.6; đây là phân tích sau khi xem kết quả giai đoạn A và được ghi như vậy.

**V2.11 — Tham số sklearn.** Với IsolationForest và HistGradientBoostingClassifier: ghi mọi tham số (`get_params()`), `random_state`, phiên bản scikit-learn vào JSON kết quả. HGB dùng tham số mặc định, trong đó `early_stopping='auto'`: bật khi tập train > 10 000 mẫu; khi đó sklearn tự tách 10% **của tập train** (`validation_fraction=0.1`, ngẫu nhiên theo `random_state`) và dừng khi loss trên phần đó không cải thiện sau 10 vòng (`n_iter_no_change=10`). Không dùng val hay test của chúng tôi cho việc dừng sớm.

**V2.12 — Quy tắc cho câu hỏi 2 (khoá trước B2).** Mô hình sâu chỉ được ghi "hơn baseline" nếu, so cặp trên cùng fold PB-U r4.2, PR-AUC của nó cao hơn PR-AUC của **baseline không giám sát tốt nhất** ở ≥ 8/10 fold **và** trung bình tăng tương đối ≥ 10%. "Baseline không giám sát tốt nhất" = baseline không giám sát có PR-AUC trung bình cao nhất trên PB-U r4.2 ở B1 (không tính HGB, không tính biến thể có nhóm I; luật được tính vì không dùng nhãn). Ngoài ra ghi "không vượt".

### Chi tiết thực thi (chốt trước khi chạy, phát sinh khi viết code)

1. **G3 khi thiếu logon/logoff:** nếu ngày d không có logon thì "giờ logon đầu" = giờ của dòng log đầu tiên trong ngày (mọi file); tương tự "giờ logoff cuối" = giờ dòng log cuối. Độ dài = cuối − đầu (giờ thập phân).
2. **"Số PC khác nhau" (G4)** đếm trên cả 5 file. PC chính = PC có nhiều logon nhất trong các ngày < d; hoà thì lấy PC đạt số đó sớm nhất.
3. **Nhóm D tính trên số đếm thô** (chưa log1p), đúng công thức §3.3; σ là độ lệch chuẩn tổng thể (ddof = 0) của các user-day trước d. Đặc trưng G đưa vào mô hình ở dạng log1p (đếm) hoặc nguyên (giờ, cờ).
4. **Email:** r4.2 không có cột activity → mọi dòng email tính là Send. Số đính kèm: r4.2 là số nguyên trong cột `attachments`; r6.2 là số phần tử ngăn bởi `;`. "Kích thước trung bình" = trung bình `size` của email Send trong ngày (0 nếu không có). G1 và G2 đếm **mọi** dòng email (gồm View ở r6.2) để khớp bảng đếm giai đoạn A; G7 chỉ đếm Send.
5. **Nhóm đuôi file (G6):** đuôi = phần sau dấu chấm cuối của `filename`, viết thường. Trong dữ liệu chỉ thấy doc, pdf, txt, jpg, zip, exe.
6. **Thiết bị rời và file mồi (R, r6.2):** ghi ra/đọc từ thiết bị rời = `to_removable_media`/`from_removable_media` = True. Chạm file mồi = dòng file có (`filename`, `pc`) trùng một dòng của `decoy_file.csv`.
7. **Tập fit của mô hình:** PB-U = user-day của user **fit** (user val dùng cho τ_b và ngưỡng T2); PB-T = giai đoạn train. Isolation Forest và HGB fit một lần cho mỗi fold PB-U (`random_state` = 1000 + chỉ số lần chạy) và 5 lần cho PB-T (`random_state` = 0..4). Baseline tất định chạy 1 lần.
8. **recall@b, precision@b:** k = max(1, round(b·N)). Điểm hoà ở biên top-k được tính theo **kỳ vọng** dưới thứ tự ngẫu nhiên (quan trọng với baseline rời rạc như luật, khối lượng).
9. **Baseline ngẫu nhiên** báo giá trị giải tích (AUC 0.5, PR-AUC = tỉ lệ dương, recall@b = b).
10. **Ô hiếm:** đúng định nghĩa addendum (b): ô F38 (activity × thứ × giờ) hiếm nhất chiếm ≤ 1% số dòng train; điểm user-day = tỉ lệ dòng của user-day nằm trong ô hiếm.
11. **z-score:** hai điểm, max |z| và tổng z dương, trên 14 đặc trưng z của nhóm D.
12. **F1 ở T2:** ngưỡng tốt nhất trên val; không tính khi val không có dương (r6.2 PB-T, và các fold PB-U r6.2 mà val không có insider).
13. **Mức insider:** "phát hiện @b" = có ≥ 1 user-day dương lọt top b% của test. "Ngày tới cảnh báo" = số ngày từ ngày của `win_start` tới user-day đầu tiên của insider trong [ngày `win_start`, ngày `win_end`] có điểm ≥ τ_b; insider không có user-day nào vượt τ_b trong cửa sổ được tính "không cảnh báo" và loại khỏi trung vị (số này được báo riêng).
14. **"Vượt τ_b"** = điểm > τ_b (so sánh chặt), để với baseline rời rạc (nhiều điểm bằng nhau, ví dụ luật = 0) tỉ lệ cảnh báo trên val không vượt b. "Phát hiện @b" khi điểm hoà ở biên top-k cũng tính theo kỳ vọng (xác suất ≥ 1 user-day dương hoà được chọn).
15. **PB-U r6.2:** mỗi fold test có 1 insider (leave-one-insider-out × 2 lần lặp), nên mỗi insider xuất hiện ở 2 lần chạy; bảng từng insider in cả hai.
16. **Thêm biến thể G + D + R cho HGB ở r6.2** (spec §7 chỉ nêu cho Isolation Forest), để tham chiếu có giám sát cùng bộ đặc trưng với IF G + D + R.

---

## B2-0 — 2026-10-17, sau khi thấy kết quả B1, trước mọi lần chạy B2

B1 được duyệt ngày 2026-10-17.
- **B2-0.a và B2-0.b** làm rõ quy tắc so sánh, theo chỉ định của người dùng.
- **B2-0.c và B2-0.d được thêm SAU khi đã thấy kết quả B1 và TRƯỚC mọi lần chạy B2.** Lúc viết mục này chưa có dòng code mô hình B2 nào trong `src/`, và chưa có mô hình B2 nào được huấn luyện hay chấm điểm.
- Trước khi viết, chỉ làm ba phép đo phụ, đều bằng script tạm ở scratchpad (không thuộc dự án):
  1. đuôi phân phối đặc trưng trên giai đoạn train PB-T (≤ 2010-10-31), **không đọc nhãn**;
  2. tốc độ một bước huấn luyện trên GPU, chạy trên **tensor ngẫu nhiên**, không dùng dữ liệu dự án;
  3. RAM khi nạp bảng đặc trưng r6.2.
- Khi mâu thuẫn với §8 hoặc B0 v2, mục này thắng.
- Ký hiệu: **[tự quyết]** là lựa chọn của chúng tôi, cần người dùng duyệt; **[biết B1]** là chỗ kết quả B1 có thể đã ảnh hưởng tới lựa chọn, ghi ra để người duyệt cân nhắc.
- **B2-0 được duyệt ngày 2026-10-19, kèm điều chỉnh.** Các điều chỉnh đã sửa thẳng vào d1–d6 bên dưới; danh sách và phần đối chiếu mã tham chiếu nằm ở **B2-0.e**. Các điểm [tự quyết] không bị điều chỉnh coi như đã duyệt.

### B2-0.a — V2.12 làm rõ: cách tính kết luận

Đơn vị so sánh là 10 lần chạy PB-U r4.2, ghép cặp theo (lần lặp, fold) đúng như trong `results/phaseB/b1_r4.2_PBU.json`. Với lần chạy f:
- M_f = PR-AUC của mô hình trên test của fold (nhãn cửa sổ);
- B_f = PR-AUC của baseline trên cùng fold.

Các đại lượng:
- n_hơn = số fold có M_f > B_f (so sánh chặt); n_kém = số fold có M_f < B_f.
- **R_chính = trung bình(M) / trung bình(B)** — tiêu chí chính.
- **R_phụ = trung bình(M_f / B_f)** — luôn báo kèm.
- Kết luận được tính hai lần, một lần với R = R_chính và một lần với R = R_phụ (cùng n_hơn, n_kém). Nếu hai lần cho cùng kết luận thì ghi kết luận đó. Nếu khác nhau thì ghi **"không kết luận"**.
- Chỉ PB-U r4.2 quyết định kết luận. PB-T r4.2, PB-U r6.2 và PB-T r6.2 được báo cùng dạng bảng (n_hơn, R_chính, R_phụ) nhưng chỉ để mô tả.

**Giá trị mốc từ B1 (khoá; thứ tự lần chạy 0–9 = lần lặp 0 fold 0–4, rồi lần lặp 1 fold 0–4):**

| Mốc | Lần chạy 0–9 | Trung bình |
|---|---|---|
| z-score tổng z dương (`zpos`) | 0.0492, 0.0399, 0.0302, 0.0639, 0.0283, 0.0372, 0.0336, 0.0450, 0.0468, 0.0433 | 0.0418 → R_chính ≥ 1.10 ⇔ trung bình(M) ≥ 0.0460 |
| HGB G+D (`HGB_GD`) | 0.6990, 0.7350, 0.4325, 0.5020, 0.4940, 0.6861, 0.6401, 0.6719, 0.5584, 0.4377 | 0.5857 |

Các số trên làm tròn 4 chữ số; khi tính dùng giá trị đầy đủ trong JSON.

### B2-0.b — Deep SVDD và Temporal AE (không dùng nhãn)

So với baseline không giám sát tốt nhất (`zpos`, mốc 0.0418). Có hai kết cục:
- **"hơn baseline"** khi n_hơn ≥ 8 **và** R ≥ 1.10;
- **"không vượt"** trong mọi trường hợp còn lại;
- và **"không kết luận"** khi R_chính và R_phụ cho hai kết luận khác nhau (B2-0.a).

### B2-0.c — Deep SAD (dùng nhãn)

**So với HGB G+D** (cùng nhãn: mọi user-day dương của tập fit), có ba kết cục đối xứng:
- **"hơn"**: n_hơn ≥ 8 **và** R ≥ 1.10;
- **"kém"**: n_kém ≥ 8 **và** R ≤ 0.90;
- **"ngang"**: mọi trường hợp còn lại;
- **"không kết luận"** khi R_chính và R_phụ cho hai kết luận khác nhau.

HGB là tham chiếu **không tinh chỉnh**: tham số mặc định, tự dừng ở 10–11 vòng. Vì vậy các kết cục trên chỉ là so với tham chiếu này, không phải so với một mô hình có giám sát tốt nhất.

Deep SAD **vẫn được báo so với `zpos`**, theo đúng quy tắc B2-0.b. Kết cục này ghi riêng, có kèm chú thích "mô hình dùng nhãn, baseline thì không".

### B2-0.d — Chi tiết chạy (chốt trước khi chạy)

#### d1. Đầu vào, chuẩn hoá và đuôi nặng

- **Bộ đặc trưng:**
  - r4.2: G+D (88 cột).
  - r6.2: G+D (88 cột) là bộ chính. G+D+R (98 cột) là biến thể **chỉ cho Deep SVDD và Deep SAD**; Temporal AE ở r6.2 chỉ dùng G+D. **[tự quyết, biết B1]** Lý do là chi phí (Temporal AE r6.2 tốn nhiều giờ, xem d7). Ở B1, nhóm R không giúp IF; điều này có thể đã ảnh hưởng tới quyết định không chạy biến thể R cho Temporal AE.
  - Không dùng nhóm I.
- **Ma trận:** `userday.matrix`, giống B1: log1p cho cột đếm và `g7_size_mean`; giờ, cờ và nhóm D để nguyên.
- **Chuẩn hoá:** mean và std (ddof = 0) ước lượng **chỉ trên hàng fit** (PB-U: user fit của fold; PB-T: giai đoạn train), rồi áp cho mọi hàng. Cột nào có std < 1e-6 trên tập fit thì đặt = 0 ở mọi hàng (coi như bỏ) và ghi tên vào JSON. Mục đích là tránh chia cho ~0 khi val/test có giá trị khác.
- **Đuôi nặng, số đo trên giai đoạn train PB-T, không nhãn:**

  | | \|s\| phân vị 99% | 99.9% | max | hàng có ít nhất một \|s\| > 10 |
  |---|---|---|---|---|
  | r4.2 | 3.85 | 8.66 | 299 (`g8_dom_new_org`) | 2.6% |
  | r6.2 | 3.88 | 7.83 | 660 (`g1_http_k0`) | 2.5% |

  Ở đây s là giá trị sau chuẩn hoá. Riêng z thô của nhóm D có |z| max là 400 (r4.2) và 300 (r6.2).
- **Quyết định: không cắt cứng.** Sau chuẩn hoá, áp phép nén có dấu **s′ = sign(s) · log1p(|s|)** cho **mọi** cột (G, D, R), cả ở fit, val và test.
  - Phép nén đơn điệu theo từng cột, nên giữ nguyên thứ tự và không tạo điểm hoà. Nó giới hạn |s′| ≤ ~6.5 với |s| ≤ 660.
  - Lý do chính là ổn định số học: không nén thì loss MSE của AE và thống kê BN bị vài ô cỡ 100–600σ chi phối.
  - **[tự quyết, biết B1]** Chúng tôi biết từ phân tích đơn biến sau B1 (trên tập fit) rằng đuôi cực trị của `z_g8_http` mang tín hiệu. Cắt cứng ở ±k sẽ làm các giá trị đó hoà nhau. Biết điều này nên lựa chọn "nén thay vì cắt" không hoàn toàn mù.
- **Biến thể độ nhạy (đã duyệt 2026-10-19):** cắt cứng s về [−10, 10] thay cho phép nén, chỉ cho Deep SVDD ở PB-U r4.2. Đây **chỉ là phân tích độ nhạy**: báo cáo bất kể kết quả, không dùng để chọn cấu hình chính, không dùng cho kết luận. Mọi chi tiết khác (AE pretrain riêng trên dữ liệu đã cắt, chọn rep theo MSE val của AE đó, seed) giống bản chính.
- Baseline B1 (z thô) giữ nguyên, không bị ảnh hưởng.

#### d2. Mạng MLP (Deep SVDD, Deep SAD, AE pretrain)

- **Encoder φ:** `Linear(d, 64, bias=False) → BatchNorm1d(64, eps=1e-4, affine=False) → LeakyReLU(0.1) → Linear(64, rep, bias=False)`.
  - BatchNorm không affine, đặt trước activation. Không có bias ở bất kỳ lớp nào của encoder (Ruff 2018/2020: bias hoặc BN affine cho phép nghiệm sụp về tâm).
  - rep ∈ {16, 32}.
- **Decoder** (chỉ dùng khi pretrain): `Linear(rep, 64, bias=False) → BatchNorm1d(64, eps=1e-4, affine=False) → LeakyReLU(0.1) → Linear(64, d, bias=False)`. Không có activation ở đầu ra. **[tự quyết]** Bản của Ruff dùng sigmoid vì dữ liệu được min-max về [0, 1]; dữ liệu của chúng tôi đã chuẩn hoá và nén nên không bị chặn trong [0, 1].
- **BN:** khi train dùng thống kê của batch. Khi eval (tính tâm c, chấm điểm) dùng running statistics, momentum 0.1 (mặc định PyTorch).
- **Khởi tạo:** mặc định PyTorch (kaiming uniform), theo seed ở d5.
- **Nguồn:** khớp các điểm của bản Deep SAD đã dựng lại (`C:\Users\binh5\Downloads\results.md`): BN không affine, LeakyReLU 0.1, weight decay 1e-6 ở cả hai giai đoạn, AE pretrain cùng lịch với giai đoạn chính. Kết quả đối chiếu với mã tham chiếu `deepsad_bn2.py` và với mã công khai của Ruff: xem **B2-0.e**.

#### d3. Huấn luyện MLP

- **AE pretrain** (chung cho Deep SVDD và Deep SAD của cùng lần chạy và cùng rep):
  - học trên **mọi hàng fit**, không dùng nhãn;
  - loss MSE trung bình theo đặc trưng và batch;
  - Adam, lr 1e-4, weight_decay 1e-6, 150 epoch, batch 256, xáo lại mỗi epoch;
  - `drop_last=True` **[tự quyết]**, để tránh batch BN quá nhỏ;
  - không giảm lr theo mốc, không dừng sớm; dùng trọng số của epoch cuối.
  - **Lệch khỏi giao thức gốc (ghi rõ theo yêu cầu duyệt):** giao thức gốc (Ruff; mã tham chiếu) dùng lr 1e-4 trong 50 epoch đầu rồi 1e-5 trong 100 epoch còn lại. Ở đây lr **cố định 1e-4 × 150 epoch, không giảm theo mốc**, cho cả AE pretrain, Deep SVDD và Deep SAD.
- **Tâm c:** sau pretrain, chạy encoder ở chế độ eval và lấy trung bình φ(x) trên mọi hàng fit.
  - **Quy tắc đẩy tâm (sửa 2026-10-19, đúng quy tắc của Ruff):** mọi thành phần có |c_j| < 0.1 đặt **c_j = 0.1 · sign(c_j)**; c_j = 0 đúng bằng 0 thì đặt +0.1.
  - Sau đó c cố định. Deep SVDD và Deep SAD của cùng (lần chạy, rep) dùng chung c.
- **Deep SVDD:**
  - encoder khởi tạo từ encoder đã pretrain (trọng số và running statistics);
  - loss = trung bình theo batch của ‖φ(x) − c‖², với bình phương khoảng cách là tổng theo các chiều rep;
  - Adam, lr 1e-4, weight_decay 1e-6, 150 epoch, batch 256; các quy tắc khác như pretrain;
  - điểm = ‖φ(x) − c‖² ở chế độ eval.
- **Deep SAD:**
  - cùng encoder khởi tạo và cùng c như Deep SVDD;
  - loss từng mẫu: hàng không nhãn dùng ‖φ(x) − c‖²; hàng có nhãn dương dùng η · (‖φ(x) − c‖² + ε)⁻¹, với **ε = 1e-6**; lấy trung bình theo batch. Không có hàng "có nhãn bình thường";
  - lịch huấn luyện như Deep SVDD; điểm = ‖φ(x) − c‖².
  - **Nhãn:** mọi user-day fit có `y_win` = 1 là có nhãn dương (đúng tập nhãn mà HGB đã dùng); mọi user-day fit còn lại là không nhãn.
  - **γ_l** = số hàng có nhãn / số hàng fit, báo cho từng lần chạy. Dự kiến:
    - r4.2 PB-U: 749–1 024 trên ~211 nghìn, tức 0.35–0.48%;
    - r4.2 PB-T: 778 / 209 182 = 0.37%;
    - r6.2 PB-U: 9–92 trên ~892 nghìn, tức 0.001–0.010%.
  - Không lấy mẫu lại (oversampling): batch ngẫu nhiên như Ruff. Ở r4.2, trung bình khoảng 1 hàng có nhãn mỗi batch.
- **Lưới và cách chọn (không dùng test):**
  - **rep ∈ {16, 32}:** chọn cho từng lần chạy theo **MSE tái tạo của AE pretrain trên val** (chế độ eval). Tiêu chí này không dùng nhãn và so sánh được giữa hai giá trị rep. Nếu chênh tương đối < 1% thì chọn 16. Rep đã chọn dùng cho cả Deep SVDD và Deep SAD của lần chạy đó. **[tự quyết]** §8 nói "loss trên val", nhưng loss SVDD không so sánh được giữa hai số chiều rep. Vì vậy, chỉ cần huấn luyện Deep SVDD và Deep SAD ở rep đã chọn.
  - **η ∈ {0.5, 1, 2}** (Deep SAD):
    - r4.2, cả PB-U và PB-T: chọn cho từng lần chạy theo **PR-AUC trên val** (nhãn cửa sổ). Val PB-T r4.2 có 226 dương. Hoà thì ưu tiên η gần 1, rồi η nhỏ hơn.
    - r6.2 PB-U: **chạy Deep SAD với η = 1 cố định**, vì val chỉ có 1 insider (§8). (Xác nhận khi duyệt 2026-10-19.)
    - r6.2 PB-T: **Deep SAD KHÔNG chạy** (§8: train chỉ có 9 dương). (Xác nhận khi duyệt 2026-10-19.)
  - **Bộ G+D+R** (r6.2) chỉ dùng cho Deep SVDD và Deep SAD; Temporal AE chỉ chạy G+D. (Xác nhận khi duyệt 2026-10-19.)
- **Đọc test:** với mỗi lần chạy, test chỉ được chấm cho **cấu hình đã chọn**. Các mô hình không được chọn chỉ được chấm trên val rồi bỏ. Như vậy test được đọc đúng một lần cho mỗi cấu hình đã chốt.
- **Không dừng sớm** ở bất kỳ mô hình nào. Loss trên val vẫn được ghi theo từng epoch để báo cáo. **[tự quyết]** §5 cho phép mô hình không giám sát ở PB-T r6.2 dừng sớm theo loss val; chúng tôi dùng số epoch cố định ở mọi nơi cho đồng nhất.

#### d4. Temporal AE

- **Cửa sổ của ngày d (user u):** gọi a_1 < a_2 < … là các ngày hoạt động của u (các user-day có trong bảng), và d = a_k. Cửa sổ là [x(a_{k−W+1}), …, x(a_k)], tức **ngày d cùng W−1 ngày hoạt động liền trước của chính user**, bất kể khoảng cách theo lịch.
  - Nếu u có ít hơn W−1 ngày hoạt động trước d, các vị trí đầu cửa sổ được đệm vector 0 và mặt nạ đặt bằng 0.
  - Nhờ vậy **mọi user-day đều có cửa sổ**, và mọi mô hình chấm đúng cùng một tập user-day như B1.
  - Vector 0 ứng với trung bình của tập fit trong không gian đã chuẩn hoá. Mặt nạ chỉ dùng trong loss và điểm, không đưa vào mạng như một kênh **[tự quyết]**.
- **Chỉ dùng quá khứ:** cửa sổ chỉ chứa ngày ≤ d của cùng user.
  - PB-T: cửa sổ của ngày val/test được phép chứa ngày của giai đoạn trước, vì đó là quá khứ (§5). Cửa sổ của ngày fit tự động chỉ chứa ngày fit.
  - PB-U: mỗi user chỉ thuộc một tập, nên cửa sổ không vượt qua ranh giới giữa các tập.
- **Đầu vào:** chuẩn hoá và nén như d1, với bộ chuẩn hoá fit trên hàng fit.
- **Mạng:**
  - encoder `GRU(d → 64, 1 lớp)`; z = trạng thái ẩn cuối (64 chiều);
  - decoder `GRU(64 → 64)`, nhận z lặp lại W lần làm đầu vào, rồi `Linear(64 → d)` ở từng bước;
  - Linear đầu ra có bias **[tự quyết]**, vì đây không phải mô hình siêu cầu;
  - tái tạo theo thứ tự xuôi.
- **Huấn luyện:**
  - mỗi user-day fit cho một cửa sổ, tức một mẫu;
  - loss = MSE có mặt nạ, trung bình trên các vị trí hợp lệ và các đặc trưng;
  - Adam, lr 1e-3, weight_decay 0 **[tự quyết]**, 50 epoch, batch 256, xáo lại mỗi epoch, `drop_last=True`;
  - cắt chuẩn gradient ở 1.0 **[tự quyết]**;
  - dùng trọng số của epoch cuối, không dừng sớm.
- **Điểm của ngày d** = MSE (trung bình theo đặc trưng) ở **vị trí cuối** (chính ngày d) của cửa sổ kết thúc tại d, ở chế độ eval.
- **W ∈ {7, 14}:** chọn cho từng lần chạy theo trung bình MSE ở vị trí cuối trên **val**. Đây đúng là đại lượng dùng làm điểm, không dùng nhãn, và so sánh được giữa hai giá trị W. Nếu chênh tương đối < 1% thì chọn 7.

#### d5. Seed và tính tất định

- **Seed:**
  - PB-U: lần chạy i (0–9, cùng thứ tự với JSON B1) dùng seed 1000 + i;
  - PB-T: seed 0–4;
  - mọi điểm lưới trong cùng một lần chạy dùng chung seed.
- **Seed điều khiển gì:** khởi tạo trọng số và thứ tự xáo batch (mỗi epoch một hoán vị). Mỗi mô hình có `torch.Generator` riêng trên CPU, khởi tạo bằng seed của nó. Vì vậy kết quả của một mô hình không phụ thuộc việc nó được xếp chồng cùng mô hình nào, ngoài sai khác dấu phẩy động của phép nhân ma trận theo lô.
- **Thiết lập tất định:**
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`;
  - `cudnn.benchmark=False`, `cudnn.deterministic=True`;
  - tắt TF32 (`torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`); float32, không AMP, không dropout;
  - dữ liệu nằm sẵn trên GPU.
- **Ghi vào JSON:** phiên bản torch, CUDA, cuDNN, tên GPU, driver.
- **Phạm vi tái lập:** chỉ đảm bảo trên cùng máy, cùng phần mềm và cùng cách nhóm xếp chồng (d6).
- **Kiểm thử trước khi chạy thật** (không đọc test):
  - **(f)** chạy AE pretrain và Deep SVDD của lần chạy 0, PB-U r4.2, mỗi giai đoạn 2 epoch, **hai lần**, trên GPU với `torch.use_deterministic_algorithms(True)` và `CUBLAS_WORKSPACE_CONFIG=:4096:8`; điểm val phải giống hệt từng bit. Temporal AE (W = 7, 1 epoch) được kiểm tra cùng cách. Nếu không đạt từng bit thì ghi **"tái lập xấp xỉ"** kèm sai khác lớn nhất đo được, không che đi.
  - **(g)** bản MLP xếp chồng (d6) phải khớp bản `nn.Module` tham chiếu (`nn.Linear`, `nn.BatchNorm1d`, `torch.optim.Adam`) huấn luyện riêng từng mô hình: loss, trọng số, running statistics của BN, tâm c và điểm, trên dữ liệu nhỏ, CPU, float64, sai số ≤ 1e-10. Test phải **chứng minh các mô hình trong chồng độc lập với nhau** về:
    - batch: mỗi mô hình có tập fit và dòng chỉ số riêng (các tập fit có kích thước khác nhau, để kiểm tra cả việc đóng băng mô hình xong trước);
    - seed: kết quả của một mô hình không đổi khi đổi seed hoặc dữ liệu của mô hình khác trong chồng, khi đổi thứ tự trong chồng, và khi chạy một mình (K = 1);
    - thống kê BN: running mean/var của từng mô hình khớp bản tham chiếu chạy riêng.
  - **(h)** bộ dựng cửa sổ Temporal AE:
    - cửa sổ chỉ chứa ngày ≤ d của cùng user;
    - đệm và mặt nạ đúng;
    - mọi user-day đều có cửa sổ;
    - đổi dữ liệu của các ngày > d thì cửa sổ và điểm của ngày d không đổi (tương tự test a).
  - **(i)** loss Deep SAD và quy tắc đẩy tâm ±eps khớp công thức trên một batch dựng tay.
  - **(j)** chỉ số nhãn dương mà Deep SAD dùng là tập con của tập fit (và đúng bằng các hàng fit có y_win = 1); nhãn val và test **không được đọc** trước khi chấm điểm lần cuối. Nhãn nằm sau một "cổng": hàng fit đọc được; val chỉ mở sau khi mọi mô hình của nhóm đã huấn luyện xong; test chỉ mở sau khi cấu hình đã được chọn. Đọc khi cổng còn đóng thì báo lỗi. Test kiểm tra cả thứ tự này trên dữ liệu tổng hợp, và kiểm tra test chỉ được chấm cho cấu hình đã chọn.
  - **(k)** cửa sổ Temporal AE không vượt ranh giới user: mọi vị trí hợp lệ trong cửa sổ của user u là hàng của u; vị trí của user khác luôn có mặt nạ 0.
- **Chạy thử (trước lưới đầy đủ, theo yêu cầu duyệt):** lần chạy 0 PB-U r4.2, mỗi mô hình một cấu hình (AE rep 16 → Deep SVDD rep 16; Deep SAD rep 16, η = 1; Temporal AE W = 7), đủ số epoch như chạy thật, **chỉ dùng fit và val, không chấm test**. Không tính metric theo nhãn val (chỉ loss, khoảng cách đến tâm, MSE val), vì chạy thử chỉ để kiểm tra kỹ thuật. Báo: loss có giảm không, có NaN không, khoảng cách đến tâm của Deep SVDD có suy biến không, thời gian thực đo và dự báo lại tổng thời gian. Dừng sau bước này để người dùng xác nhận.

#### d6. Cách chạy và thứ tự

- **MLP xếp chồng:** K mô hình chạy trong một tiến trình, trọng số dạng [K, vào, ra] và nhân bằng `bmm`.
  - BN tính riêng cho từng mô hình (viết tay, kể cả running statistics).
  - Mỗi mô hình có dòng chỉ số batch riêng, trên tập fit của riêng nó.
  - Mỗi mô hình chạy đúng 150 epoch trên tập fit của nó. Vì các fold có số hàng khác nhau (< 0.6%), mô hình nào xong trước thì đóng băng.
- **Cách nhóm (cố định),** theo từng (bản, giao thức, bộ đặc trưng):
  - giai đoạn 1 = mọi AE của **cùng một rep** (mọi lần chạy); hai rep là hai nhóm, vì trọng số xếp chồng phải cùng kích thước (sửa khi viết code, 2026-10-19);
  - giai đoạn 2 = mọi Deep SVDD và Deep SAD (lần chạy × η) đã chọn **cùng một rep**;
  - tối đa 40 mô hình mỗi nhóm; nếu vượt thì tách theo thứ tự lần chạy.
- **Chuẩn hoá trong chồng:** ma trận thô nằm một lần trên GPU; mỗi mô hình mang mean/std/cột bỏ của riêng lần chạy của nó và chuẩn hoá + nén từng batch khi lấy ra. Kết quả giống hệt việc chuẩn hoá trước, nhưng không cần K bản sao ma trận (r6.2: 10 × 0.5 GB).
- **Deep SVDD trong cùng chồng với Deep SAD:** Deep SVDD dùng đúng loss Deep SAD với tập nhãn rỗng, nên công thức trùng với d3.
- **Lưu trọng số:** mỗi bước huấn luyện lưu trọng số, running statistics, tâm c và tham số chuẩn hoá vào `results/phaseB/b2_ckpt/` (không đưa vào báo cáo). Bước chấm điểm riêng đọc lại các file này: mở nhãn val → chọn η → mở nhãn test → chấm test cho cấu hình đã chọn.
- **Temporal AE:** mỗi (lần chạy, W) là một tiến trình. Tối đa 4 tiến trình GPU cùng lúc, tính cả tiến trình MLP.
- **Thứ tự:**
  1. r4.2 PB-U: MLP giai đoạn 1 → giai đoạn 2 → Temporal AE;
  2. r4.2 PB-T;
  3. r6.2 PB-U;
  4. r6.2 PB-T.

  Toàn bộ quy trình đã chốt ở trên. Thứ tự chỉ có ý nghĩa nếu hết thời gian; không chỉnh gì giữa các bước theo kết quả.
- Lệnh nào dài quá 5 phút thì chạy nền, log vào `logs/phaseB2_*.log`, kèm bộ giám sát RAM (dừng nếu vượt 20 GB).
- **Đầu ra:**
  - `results/phaseB/b2_r<rel>_<proto>_<model>.json`: mỗi lần chạy có tham số đã chọn, tiêu chí val của mọi điểm lưới, γ_l, đường loss, mọi metric như B1 (gồm theo kịch bản, từng insider, τ_b từ val), thời gian và phiên bản;
  - điểm val/test của cấu hình đã chọn: `results/phaseB/scores/*.npz`;
  - bảng B2: thêm vào `results/phaseB_tables.md`; báo cáo B2: thêm một phần vào `REPORT_B.md` **[tự quyết]**.

#### d7. Ước lượng thời gian và bộ nhớ (RTX 4060 8 GB)

**Số đo** trên tensor ngẫu nhiên, batch 256, Adam, thiết lập tất định ở d5:

| Phép đo | ms/bước |
|---|---|
| MLP một mô hình: SVDD (encoder) | 2.7–3.0 |
| MLP một mô hình: AE | 4.2–4.5 |
| MLP xếp chồng, 10 AE cùng lúc | 4.5 (cho cả nhóm) |
| MLP xếp chồng, 20 AE cùng lúc | 3.9 (cho cả nhóm) |
| GRU Temporal AE, chạy một mình | ~8 |
| GRU Temporal AE, 4 tiến trình song song | 8.7–11 mỗi tiến trình |

Bước MLP chủ yếu tốn chi phí khởi chạy kernel, nên xếp chồng gần như "miễn phí". Nhóm 40 mô hình chưa đo; ước lượng 5–8 ms/bước.

**Số bước mỗi epoch:** r4.2 PB-U ~825; r4.2 PB-T ~817; r6.2 PB-U ~3 484; r6.2 PB-T ~3 294.

| Cấu hình | Mô hình huấn luyện | Ước lượng |
|---|---|---|
| r4.2 PB-U, MLP | GĐ1: 20 AE (124 nghìn bước) ≈ 9 phút; GĐ2: 10 SVDD + 30 SAD ≈ 10–17 phút | ~20–30 phút |
| r4.2 PB-U, Temporal AE | 20 lần chạy (10 × 2 W), mỗi lần 41 nghìn bước ≈ 7 phút; 4 song song | ~35–40 phút |
| r4.2 PB-T, MLP | GĐ1: 10 AE ≈ 9 phút; GĐ2: 5 SVDD + 15 SAD ≈ 8 phút | ~20 phút |
| r4.2 PB-T, Temporal AE | 10 lần chạy, 3 lượt | ~25 phút |
| r6.2 PB-U, MLP | GĐ1: 40 AE (2 bộ đặc trưng × 10 × 2 rep; 523 nghìn bước) ≈ 40–80 phút; GĐ2: 2 bộ × (10 SVDD + 10 SAD) ≈ 45–70 phút | ~1.5–2.5 giờ |
| r6.2 PB-U, Temporal AE | 20 lần chạy, mỗi lần 174 nghìn bước ≈ 30 phút; 5 lượt | ~2.5 giờ |
| r6.2 PB-T, MLP | GĐ1: 20 AE ≈ 35 phút; GĐ2: 10 SVDD ≈ 35 phút | ~1–1.5 giờ |
| r6.2 PB-T, Temporal AE | 10 lần chạy, mỗi lần 165 nghìn bước ≈ 29 phút; 3 lượt | ~1.5 giờ |

- **Tổng thời gian GPU:** ~8–10 giờ. Nếu cho MLP chạy song song với Temporal AE thì thời gian thực ~6–8 giờ. Sai số ước lượng ±50%. Chấm điểm thêm vài phút mỗi cấu hình.
- **Bộ nhớ GPU mỗi tiến trình:**
  - ma trận đặc trưng trên GPU: r4.2 ~116 MB; r6.2 G+D ~490 MB; G+D+R ~546 MB;
  - phép đo đạt đỉnh 181–450 MiB, cộng ngữ cảnh CUDA ~0.3–0.5 GB;
  - tổng ≤ ~1.2 GB mỗi tiến trình, ≤ ~5 GB cho 4 tiến trình (GPU có 8 GB).
- **RAM máy:** nạp r6.2 và dựng ma trận đo được 1.4 GiB; cộng torch thì ≤ ~3 GB mỗi tiến trình, ≤ ~12 GB cho 4 tiến trình (máy có 32 GB).

### B2-0.e — Điều chỉnh khi duyệt (2026-10-19) và đối chiếu mã tham chiếu

**Điều chỉnh của người duyệt (đã sửa vào d1–d6):**
1. Quy tắc đẩy tâm (d3): |c_j| < 0.1 → c_j = 0.1 · sign(c_j); c_j = 0 → +0.1.
2. Lịch lr (d3): ghi rõ là lệch khỏi giao thức gốc — lr cố định 1e-4 × 150 epoch, không giảm theo mốc.
3. Biến thể cắt cứng ±10 (d1): duyệt, chỉ là phân tích độ nhạy, báo bất kể kết quả, không dùng để chọn cấu hình chính.
4. Xác nhận (d3): Deep SAD không chạy ở r6.2 PB-T; chạy ở r6.2 PB-U với η = 1; G+D+R chỉ cho Deep SVDD và Deep SAD.
5. Test mới (j), (k); test (f) và (g) được nói rõ hơn (d5).
6. Chạy thử 1 fold trước lưới đầy đủ, dừng chờ xác nhận (d5).

**Mã tham chiếu `deepsad_bn2.py`: nhận ngày 2026-10-19 (sau bước chạy thử), đã đặt ở `docs/reference/deepsad_bn2.py`** (sha256 `de1851be…ee035c`, giống bản trong Downloads). Bảng đối chiếu từng dòng với file nằm ở **B2-0.f**. Bảng dưới đây là bản đối chiếu cũ, làm khi chưa có file, với các giá trị người duyệt ghi trong tin nhắn; giữ lại để lưu vết. `Downloads\paper.zip` là một mô hình khác (Temporal AE GRU + đầu SVDD), không phải mã tham chiếu.

| Điểm | Mã công khai của Ruff (theo trí nhớ, chưa mở lại mã) | Tham chiếu `deepsad_bn2.py` (theo tin nhắn duyệt) | Spec B2 | Kết quả |
|---|---|---|---|---|
| Đẩy tâm | `c[(abs(c) < eps) & (c < 0)] = -eps`; `c[(abs(c) < eps) & (c > 0)] = eps`, eps = 0.1 (c_j = 0 giữ nguyên 0) | dòng 92–93: đẩy \|c_j\| < 0.1 ra ±0.1 | 0.1 · sign(c_j); c_j = 0 → +0.1 | khớp; chỉ khác ở trường hợp c_j = 0 đúng bằng 0, người duyệt đã chốt +0.1 |
| BatchNorm | `BatchNorm1d(eps=1e-04, affine=False)` sau Linear không bias, trước activation | eps = 1e-4, affine = False, trước LeakyReLU | như vậy | khớp |
| Activation | `F.leaky_relu` **độ dốc mặc định 0.01** (nhớ, không chắc) | LeakyReLU(0.1) | LeakyReLU(0.1) | spec theo tham chiếu; có thể lệch với mã công khai |
| ε của loss Deep SAD | `eps = 1e-6`, loss có nhãn η · (dist + eps)^(−1) | 1e-6 | 1e-6 | khớp |
| Lịch lr | lr 1e-4, 150 epoch, mốc giảm ×0.1 ở epoch 50 (cả AE pretrain) | 1e-4 × 50 rồi 1e-5 × 100 | **cố định 1e-4 × 150** | **lệch, đã ghi rõ** (d3) |
| Đầu ra decoder | sigmoid (dữ liệu min-max [0, 1]) | không nêu | không activation | lệch có chủ ý (d2) |
| Batch, weight decay | README của Ruff: batch 128, weight_decay 0.5e-6, ae_weight_decay 0.5e-3 (nhớ, không chắc) | không nêu | batch 256 (§8), wd 1e-6 cả hai giai đoạn (`results.md`) | có thể lệch với mã công khai; giữ theo §8 và `results.md` |
| Tâm c tính ở | `net.eval()`, trung bình trên toàn bộ dữ liệu train | không nêu | eval, mọi hàng fit | khớp với trí nhớ |
| Điểm bất thường | dist = ‖φ(x) − c‖² | không nêu | ‖φ(x) − c‖² | khớp với trí nhớ |

### B2-0.f — Đối chiếu từng dòng với `docs/reference/deepsad_bn2.py` (2026-10-19)

File tham chiếu viết cho bộ dữ liệu ODDS (arrhythmia, cardio, …), đọc dữ liệu ở `/home/claude/deepsad/data/*.npz`. Máy này không có các file đó, nên không chạy được nguyên trạng trên chính dữ liệu của nó. Việc tải về cần ra ngoài sandbox, nên chưa làm. Script kiểm tra là `src/refcheck_b2.py`; nó import file tham chiếu, không sửa gì.

**Giống nhau, đã kiểm bằng số** (`results/phaseB/b2_refcheck_equiv.json`, float64, cùng trọng số):

| Điểm | Tham chiếu | `deep_b2.py` | Sai khác đo được |
|---|---|---|---|
| Đẩy tâm (dòng 92–93) | c<0 → −0.1, c≥0 → +0.1 khi \|c\|<0.1 | `push_center` | 0 (trên các giá trị biên và 1e5 giá trị ngẫu nhiên) |
| Lớp (dòng 36–43) | Linear không bias → BN(eps 1e-4, affine=False) → LeakyReLU(0.1); lớp cuối của encoder/decoder chỉ là Linear | giống | forward ở chế độ train 1.1e-15, ở chế độ eval 2.8e-16; running mean/var ≤ 2.2e-16 |
| Adam + weight decay L2 | `torch.optim.Adam(weight_decay=WD)` | `StackAdam` | 1.6e-15 sau 20 bước |
| Loss Deep SAD (dòng 98) | `where(s==0, d, η(d+1e-6)^s)`, s = −1 | `sad_loss` | 0 với η = 0.5, 1, 2 |
| Điểm | ‖φ − c‖ (căn) | ‖φ − c‖² | cùng thứ hạng, nên ROC/PR-AUC như nhau |
| c tính ở | `enc.eval()`, trung bình φ | eval, trung bình φ | — |

Kiểm tra này phát hiện một lỗi nhỏ: `push_center` tạo hằng 0.1 ở float32, nên khi chạy float64 lệch 1.5e-9. Lỗi đã được sửa. Các lần chạy thật dùng float32 nên kết quả không đổi. Test B2 chạy lại vẫn đạt. `one_seed()` của file tham chiếu cũng chạy được trọn vẹn trên một tập tổng hợp ở chế độ quick (AUC 0.97), nên môi trường (torch 2.8, sklearn 1.7.2) không có vấn đề.

**Khác nhau:**

| # | Điểm | Tham chiếu | Spec B2 | Ghi chú |
|---|---|---|---|---|
| 1 | Lịch lr (dòng 46–53) | Adam 1e-4 × 50, rồi **tạo Adam mới** (moment về 0) 1e-5 × 100; AE pretrain cùng lịch | 1e-4 cố định × 150, một Adam | lệch đã ghi ở d3 |
| 2 | Batch | 200, **giữ batch lẻ cuối** | 256, bỏ batch lẻ (drop_last) | batch lẻ cỡ 1 sẽ làm BN báo lỗi; dữ liệu ở đây không gặp trường hợp này |
| 3 | Loss AE (dòng 88) | tổng theo đặc trưng, trung bình theo batch | trung bình theo cả hai | lệch hệ số d (= 88, đã đo). Với Adam gần như không đổi, trừ việc weight decay 1e-6 trong spec mạnh hơn tương đối khoảng 88 lần |
| 4 | Dữ liệu cho AE pretrain và c (dòng 86, 91) | chỉ các hàng **không nhãn**, bỏ các anomaly có nhãn | mọi hàng fit; Deep SAD dùng chung AE và c với Deep SVDD | ở r4.2 run0 chênh 829/211 334 hàng (0.39 %) |
| 5 | Độ sâu | 3 lớp, theo từng bộ ODDS (ví dụ 32-16-8) | 2 lớp, d → 64 → rep (§8) | không có kiến trúc tương ứng cho CERT; giữ §8 |
| 6 | Tiền xử lý | StandardScaler (fit trên train, gồm cả anomaly) | chuẩn hoá trên fit + signed log1p (hoặc cắt ±10) | d1 |
| 7 | Tập không nhãn | γ_p = 0: bỏ các anomaly train không được gán nhãn | ở Deep SAD mọi hàng dương của fit đều có nhãn, nên tương đương | — |
| 8 | Seed | `torch.manual_seed` toàn cục | Generator riêng cho mỗi mô hình (test g) | không ảnh hưởng phép toán |
| 9 | Docstring | ghi "batch-norm omitted", nhưng code (`mlp`) **có** BN | — | mâu thuẫn trong chính file tham chiếu; ở đây theo code |

**Chạy giao thức của tham chiếu trên fold chạy thử** (`src/refcheck_b2.py proto`, `results/phaseB/b2_refcheck_proto.json`). Thiết lập là r4.2 PB-U run0, rep 16, cùng tiền xử lý và kiến trúc của spec; lặp lại dòng 83–100 của `one_seed`, tức là lịch lr hai pha, batch 200, loss AE dạng tổng, AE và c chỉ trên các hàng không nhãn. Chỉ dùng fit và val, không có metric theo nhãn val. Cổng nhãn chỉ ghi `read fit positives`.

| | Tham chiếu: SVDD | Spec (chạy thử): SVDD | Tham chiếu: SAD η=1 | Spec: SAD η=1 |
|---|---|---|---|---|
| Loss epoch 1 → 150 | 0.980 → 0.000263 | 1.022 → 0.000216 | 0.944 → 0.0130 | 1.028 → 0.0099 |
| NaN | không | không | không | không |
| MSE val của AE | 0.0176 | 0.0146 | 0.0180 | (dùng chung AE) 0.0146 |
| Thành phần c bị đẩy | 13/16 | 11/16 | 14/16 | 11/16 |
| Trung vị khoảng cách trên val lúc khởi tạo → sau huấn luyện | 3.28 → 2.2e-5 | 2.95 → 2.6e-5 | 3.17 → 1.6e-3 | 2.95 → 9.0e-4 |
| p99 / max khoảng cách trên val | 1.6e-3 / 0.026 | 1.1e-3 / 0.014 | 0.095 / 21.3 | 0.069 / 44.3 |
| std của φ theo chiều (trung bình) | 0.0026 | 0.0021 | 0.028 | 0.034 |
| Khoảng cách trên fit: có nhãn / không nhãn | — | — | 2.34 / 0.0056 | 4.05 / 0.0036 |
| Thời gian (1 mô hình, GPU) | AE 742 s + 906 s | AE 1292 s + (SVDD+SAD xếp chồng) 803 s | AE 1001 s + 934 s | — |

**Kết luận:**
- Ở Deep SVDD, khoảng cách co về c khoảng 10⁵ lần và chạy theo giao thức tham chiếu cũng xảy ra **như vậy**. Hiện tượng này không do các điểm lệch 1–4 gây ra.
- Ở Deep SAD, hai cách cho cùng một kiểu kết quả: hàng có nhãn bị đẩy xa c, hàng không nhãn co lại gần c. Các con số chênh nhau trong vòng khoảng 2 lần.
- Sau bước kiểm tra này, spec chưa đổi gì; muốn chuyển sang giao thức tham chiếu (các điểm 1–4) thì phải được người duyệt quyết định.
