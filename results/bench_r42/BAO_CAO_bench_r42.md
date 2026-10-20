# Benchmark CERT r4.2 — phương pháp của các tác giả khác, giao thức của báo cáo Hybrid

Chạy ngày 2026-10-19/20 bằng `src/bench_r42.py`. Kết quả gốc gồm:

- `per_seed.csv`: mỗi mô hình × seed một dòng.
- `summary.csv`: trung bình và độ lệch chuẩn.
- `scores_*.npz`: điểm số trên 141 140 dòng test.
- `meta_*.json`: đường cong loss và thời gian chạy.

## Giao thức

Giao thức chép theo `BAO_CAO_TONG_KET_TOAN_BO.md`, mục 3.6, 7.1 và 12.1. Mọi con số dưới đây đã được đối chiếu khớp đúng với báo cáo.

| | Ở đây | Báo cáo |
|---|---|---|
| Dữ liệu | HF `nguynn/cert-insider-threat-datasets/cert_r4.2` + `cmu_kilthub/answers` | như vậy |
| Đơn vị, nhãn | user-day; dương khi ngày đó có ít nhất một sự kiện độc hại (`y_event`, tổng 966) | 966 |
| Mốc cắt | 2010-10-01. Insider có ngày độc hại ở cả hai phía được gán về phía có nhiều ngày độc hại hơn (hoà thì về train), user-day của người đó ở phía kia bị xoá | như vậy |
| Train | 189 237 dòng, 460 dương (45 / 404 / 11), 35 insider | 189 237, 460 (45 / 404 / 11), 35 |
| Test | 141 140 dòng, 479 dương (40 / 431 / 8), 35 insider, 229 ngày | 141 140, 479 (40 / 431 / 8), 35, 229 |
| Bị xoá | 75 user-day, trong đó 27 dương | 75, 27 |
| Chỉ số | ROC-AUC; AUC-PR (average precision); rec@k = recall trong k × 229 dòng điểm cao nhất; S1–S3@25 = recall riêng từng kịch bản trong 25 × 229 dòng đầu | như vậy |

**Khác với báo cáo, phải nêu khi so:**

1. **Đặc trưng.** Ở đây dùng bộ G+D của giai đoạn B: 88 cột, sau khi bỏ 2 cột hằng còn 86. Đặc trưng độ lệch chỉ dùng lịch sử trước ngày đang xét. Bộ này **không** có OCEAN, **không** có đặc trưng từ khoá URL, và z-score không tính theo nhóm vai trò LDAP. Báo cáo dùng 90 cột của partner; hai file `features_userday.parquet` và `labels_userday.parquet` không có trên máy và không có trên HF.
2. **Đơn vị đầu vào.** Mỗi user-day là một vector độc lập, giống các mô hình tham chiếu ở mục 4.2 của báo cáo. Hybrid của báo cáo dùng cửa sổ 7 × 90.
3. **Tiền xử lý.** signed log1p, rồi z-score với μ, σ của các dòng bình thường trong kỳ train (theo mục 3.5).

## Mô hình

Mọi siêu tham số được cố định trước khi đọc nhãn test và không tinh chỉnh gì. Mỗi mô hình chạy 5 seed (0–4); nhãn test chỉ được đọc một lần, ở bước `report`.

| Mô hình | Nguồn | Cấu hình |
|---|---|---|
| Pantelidis AE | Pantelidis et al. 2021 | d-100(tanh)-50(relu)-50(relu) và 50-100(relu)-d (tuyến tính); MSE; Adam 1e-3; 30 epoch; batch 256; học trên dòng bình thường kỳ train; điểm = MSE tái tạo |
| Pantelidis VAE | như trên | d-100(tanh)-50(relu)-latent 2; loss = tổng bình phương sai số + KL; NAdam 1e-3; 1000 epoch; batch 128; điểm = MSE của decode(μ) |
| Ruff Deep SVDD | Ruff et al. (ICML 2018), code tham chiếu `docs/reference/deepsad_bn2.py` | MLP không bias d-64-32, BN(affine=False, eps 1e-4) + LeakyReLU(0.1); AE pretrain rồi SVDD; mỗi pha Adam 1e-4 × 50 epoch, sau đó Adam mới 1e-5 × 100 epoch; batch 200; weight decay 1e-6; đẩy tâm \|c_j\| < 0.1 ra ±0.1; điểm = ‖φ − c‖² |
| Ruff Deep SAD | Ruff et al. (ICLR 2020), cùng code tham chiếu | như Deep SVDD, η = 1, gắn nhãn đủ 460 user-day dương của kỳ train |

**Lệch khỏi bài gốc:**
- Pantelidis dùng đầu ra sigmoid và BCE trên one-hot 38 chiều; ở đây dùng đầu ra tuyến tính và MSE vì đầu vào liên tục.
- Ruff dùng kiến trúc 3 lớp, chọn theo từng bộ ODDS; ở đây dùng 2 lớp d-64-32, với 32 bằng số chiều latent của Hybrid.
- Bản xếp chồng bỏ batch lẻ cuối, còn code tham chiếu giữ lại. Phép toán còn lại đã được kiểm khớp đến 1e-15 (`b2_refcheck_equiv.json`).

## Kết quả — trung bình 5 seed

| cấu hình | ROC-AUC | AUC-PR | rec@5 | rec@10 | rec@25 | rec@50 | S1@25 | S2@25 | S3@25 |
|---|---|---|---|---|---|---|---|---|---|
| Pantelidis AE, không nhãn bất thường | 0,9791 | 0,4602 | 0,6413 | 0,7144 | 0,8225 | 0,9040 | 0,965 | 0,8060 | 1,000 |
| Pantelidis VAE, không nhãn bất thường | 0,9716 | 0,1462 | 0,3624 | 0,5599 | 0,7775 | 0,9019 | 0,920 | 0,7601 | 1,000 |
| Ruff Deep SVDD, không nhãn bất thường | 0,8999 | 0,0922 | 0,2718 | 0,3958 | 0,5854 | 0,7090 | 0,850 | 0,5564 | 0,825 |
| **Ruff Deep SAD, η=1, 460 mẫu thật** | **0,9986** | **0,9124** | 0,9653 | 0,9816 | 0,9925 | 0,9975 | 0,965 | 0,9968 | 0,900 |

**Độ lệch chuẩn qua 5 seed:**

| cấu hình | ROC-AUC | AUC-PR | rec@5 | rec@10 | rec@25 | rec@50 | S1@25 | S2@25 | S3@25 |
|---|---|---|---|---|---|---|---|---|---|
| Pantelidis AE | 0,0033 | 0,0370 | 0,0306 | 0,0272 | 0,0312 | 0,0198 | 0,022 | 0,0351 | 0,000 |
| Pantelidis VAE | 0,0036 | 0,0131 | 0,0398 | 0,0307 | 0,0416 | 0,0241 | 0,011 | 0,0467 | 0,000 |
| Ruff Deep SVDD | 0,0123 | 0,0274 | 0,0690 | 0,0909 | 0,0660 | 0,0432 | 0,031 | 0,0696 | 0,209 |
| Ruff Deep SAD | 0,0007 | 0,0122 | 0,0107 | 0,0117 | 0,0043 | 0,0009 | 0,014 | 0,0035 | 0,056 |

**AUC-PR theo từng seed:**

| seed | Pantelidis AE | Pantelidis VAE | Ruff Deep SVDD | Ruff Deep SAD |
|---|---|---|---|---|
| 0 | 0,4725 | 0,1336 | 0,1005 | 0,9160 |
| 1 | 0,4326 | 0,1453 | 0,0836 | 0,8958 |
| 2 | 0,4915 | 0,1416 | 0,1334 | 0,9038 |
| 3 | 0,4106 | 0,1682 | 0,0592 | 0,9219 |
| 4 | 0,4939 | 0,1421 | 0,0842 | 0,9244 |

Mức ngẫu nhiên của AUC-PR = 479 / 141 140 = 0,0034. Cột S3 chỉ có 8 ngày dương, nên 0,900 tương đương trung bình 7,2/8.

## Đặt cạnh số của báo cáo (chỉ để tham khảo, khác bộ đặc trưng)

| | AUC-PR | ROC-AUC | rec@25 |
|---|---|---|---|
| Báo cáo — Hybrid A (Deep SVDD trên cửa sổ, 0 nhãn), 1 seed | 0,0696 | 0,9356 | 0,6681 |
| Báo cáo — Autoencoder dạng bảng, 1 seed | 0,0708 | 0,9508 | 0,6910 |
| Báo cáo — Hybrid + Deep SAD, 460 mẫu thật, 1 seed | 0,8585 | 0,9983 | 0,9916 |
| Báo cáo — Random Forest, 1 seed | 0,9302 | 0,9962 | 0,9937 |
| Ở đây — Pantelidis AE (5 seed) | 0,4602 ± 0,0370 | 0,9791 | 0,8225 |
| Ở đây — Ruff Deep SVDD (5 seed) | 0,0922 ± 0,0274 | 0,8999 | 0,5854 |
| Ở đây — Ruff Deep SAD (5 seed) | 0,9124 ± 0,0122 | 0,9986 | 0,9925 |

Hai bên dùng cùng nhãn, cùng cách chia và cùng công thức chỉ số, nhưng **khác bộ đặc trưng**. Vì vậy hai khối số này không phải một phép so công bằng giữa các mô hình. Riêng chênh lệch của Pantelidis AE (0,46 so với 0,07 của AE dạng bảng trong báo cáo) nhiều khả năng đến từ đặc trưng. Báo cáo của bạn (mục 4.4) cũng thấy điều này: chỉ cần bỏ các cột từ khoá, AE đã tăng từ 0,0708 lên 0,2251. Quy tắc "hơn" (ít nhất 4/5 seed và tỉ số ≥ 1,10) không áp dụng được, vì phía báo cáo chỉ có một seed.
