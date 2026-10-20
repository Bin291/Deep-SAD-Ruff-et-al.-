# Bước 5 — Giao thức chốt trước khi chạy (pre-registration)

Viết ngày 2026-10-16, **trước** khi huấn luyện bất kỳ mô hình nào và trước khi thấy kết quả test nào.
Mọi lựa chọn dưới đây cố định; không chỉnh theo kết quả test. Chỗ nào bài không nêu thì đánh dấu **[tự chọn]**.

## 1. Cấu hình
- **Chính:** {r4.2, r6.2} × {P1, P2, P3} × {AE, VAE} × {R1, R2}; bộ insider `main`; đặc trưng F38.
- **Biến thể (từng cái một, mọi thứ khác giữ như cấu hình chính):**
  - V30a, V30b (chỉ r4.2): bộ insider chỉ gồm kịch bản 1 / kịch bản 2; insider khác coi là bình thường. Chạy P1, P1-bal, P2, P3. **V30a × P1-bal chỉ đạt 28.4% dương** (người dùng đã duyệt giữ nguyên) vì kịch bản 1 chỉ có 775 dòng device / 818 dòng file dương, ít hơn 1667 dòng/file cần.
  - V50 (r4.2, r6.2): F38 + one-hot tháng (12) = 50 chiều. **Phỏng đoán của chúng tôi**, bài không nhắc tháng. Chạy P1, P2, P3.
  - P1-bal (r4.2, r6.2): mẫu P1 có 1/3 dòng dương mỗi file (giao thức thứ tư, chạy với bộ insider main và V30a/V30b).
- **Số lần chạy mỗi cấu hình:** 10. P1/P1-bal/P2: seed 0–9 (seed quyết định mẫu, chia và khởi tạo). P3: 2 lần lặp × 5 fold.

## 2. Dữ liệu và nhãn theo giao thức
| | Đơn vị đánh giá | Nhãn | Chia |
|---|---|---|---|
| P1 | dòng (mẫu 5000 dòng/file, Bước 3) | mức user (như bài) | 75/25 theo dòng (hoán vị chính xác) |
| P1-bal | dòng (mẫu 1/3 dương, rút từ bảng đếm) | mức user | 75/25 theo dòng |
| P2 | dòng (toàn bộ) | mức user | 75/25 theo dòng (chia nhị thức chính xác trên bảng đếm) |
| P3 | **user-day** (chính) và **user** | user-day dương nếu có ≥ 1 dòng trong cửa sổ hợp; user dương nếu là insider | theo user |

- **Validation (cho ngưỡng T2):** 20% của phần train, theo dòng ở P1/P1-bal/P2; theo user (phân tầng theo insider, ≥ 1 insider nếu train có ≥ 2) ở P3. Mô hình chỉ học trên phần "fit" (80% còn lại của train). **[tự chọn]**
- **P3 r4.2:** 5-fold phân tầng theo insider, 2 lần lặp (seed fold 0, 1). **P3 r6.2:** leave-one-insider-out, mỗi fold = 1 insider + 1/5 số user bình thường (chia ngẫu nhiên), 2 lần lặp. Mỗi fold test r6.2 chỉ có **1 insider**.
- Assert: tập user fit / val / test rời nhau ở P3.

## 3. Mô hình [kiến trúc theo bài; phần còn lại tự chọn]
- **AE:** d → 100 (tanh) → 50 (relu) → 50 (relu, bottleneck) → 50 (relu) → 100 (relu) → d (sigmoid). MSE, Adam lr 1e-3, 30 epoch, batch 256. Khởi tạo Glorot-uniform, bias 0 (mặc định Keras).
- **VAE:** encoder d → 100 (tanh) → 50 (relu) → μ, log σ² (latent 2); decoder 2 → 50 (relu) → 100 (relu) → d (sigmoid). Loss = BCE (cộng theo chiều) + KL, Nadam lr 1e-3, 1000 epoch, batch 128. Không dùng LRFinder (lr 1e-3 = mặc định Keras).
- **R1** (đọc kiểu phát hiện bất thường): học chỉ trên dòng **bình thường** của phần fit (nhãn của giao thức đó). Điểm = lỗi tái tạo: AE = MSE trung bình theo chiều; VAE = BCE cộng theo chiều giữa x và decoder(μ).
- **R2** (đọc kiểu "keep only the encoder" + "labelled data"): pretrain AE/VAE như trên trên **mọi** dòng fit (không nhãn), bỏ decoder, gắn lớp Dense(1, sigmoid) lên bottleneck (AE: 50 chiều; VAE: μ, 2 chiều), fine-tune encoder + đầu phân loại bằng BCE có nhãn, **30 epoch**, cùng optimizer và batch của mô hình. Không cân bằng lớp. Điểm = xác suất dự đoán.
- **Epoch:** đầu vào chỉ có ≤ 1176 (F38) / 14 112 (F50) vectơ phân biệt, nên mỗi lần chạy được biểu diễn bằng trọng số (số dòng) trên các ô. Mỗi minibatch rút **có hoàn lại** theo trọng số đó, tương đương rút dòng ngẫu nhiên đều. **1 epoch = 15 000 dòng rút** (= cỡ phần fit của P1: 25 000 × 0.75 × 0.8) cho **mọi** giao thức. Ở P2/P3 đây là quy đổi: cùng số bước gradient như P1, nhưng mỗi batch lấy từ phân phối của toàn bộ dữ liệu.
- 10 lần chạy của một cấu hình được huấn luyện song song như 10 mô hình độc lập (trọng số xếp chồng, Adam/Nadam theo phần tử nên tương đương tối ưu riêng).

## 4. Ngưỡng và metric
- **T1:** ngưỡng = phân vị (1 − tỉ lệ dương của train) của điểm trên toàn bộ train (fit + val), cùng đơn vị đánh giá.
- **T2:** ngưỡng cho F1 cao nhất trên validation, cùng đơn vị đánh giá.
- Dự đoán dương khi điểm ≥ ngưỡng.
- **Metric chính — "lớp độc hại (chuẩn)":** P = TP/(TP+FP), R = TP/(TP+FN), Acc, F1 = 2PR/(P+R).
- **"Weighted-avg (chẩn đoán)":** P/R/F1 từng lớp, lấy trung bình có trọng số theo số mẫu (như `classification_report` của sklearn). Chỉ để đối chiếu cách bài có thể đã tính.
- Thêm ROC-AUC và PR-AUC (không phụ thuộc ngưỡng). Báo số mẫu dương / số user dương mỗi tập test.

## 5. P3: gộp điểm (chốt trước, báo cả hai)
- **Điểm user-day:** (a) **max** điểm các dòng trong user-day; (b) **mean** điểm các dòng (trung bình theo số dòng).
- **Điểm user:** max điểm user-day của user đó (với từng quy tắc a/b).
- T1/T2 tính lại ở đúng đơn vị (user-day hoặc user) từ train/val user.

## 6. Baseline và trần đi kèm mọi bảng
- "Gắn cờ không ai" (cả hai họ metric), "gắn cờ tất cả", ngẫu nhiên (AUC 0.5), tần suất nghịch của ô, bảng tra có giám sát (tỉ lệ ô học từ train).
- Trần Bayes (oracle, in-sample) như Bước 4. Với P3: trần = điểm dòng tối ưu theo oracle (tỉ lệ dương thật của ô trên chính test) rồi gộp max/mean — đây là **trần xấp xỉ** cho mọi mô hình "chấm dòng rồi gộp", không phải cận trên chặt cho mọi bộ chấm user-day.
