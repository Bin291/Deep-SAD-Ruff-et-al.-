# Hướng dẫn tự chạy H3: AE / VAE kiểu Pantelidis trên dữ liệu của bạn

## 0. Hiện đã có kết quả nào để so?

Bảng kế hoạch (file xlsx) **chưa chứa kết quả mới nào**. "Chưa chạy" chỉ nghĩa là chưa có lần chạy nào. Thứ có sẵn để so ngay hôm nay:

| Nguồn | Kết quả | So được với Hybrid của bạn không |
|---|---|---|
| Báo cáo của bạn | Hybrid A 0.0696, Hybrid B 0.8585, AE dạng bảng 0.0708, RF 0.9302 (AUC-PR, một seed) | Đây là số của bạn |
| Giai đoạn A (Claude Code) | Tái hiện Pantelidis theo đúng bài (dòng log, one-hot 38 chiều): F1 lớp độc hại 8–17% so với 92–94% của bài; trần Bayes 9.4% trên r4.2 | **Không**: khác đơn vị (dòng log), khác chỉ số (F1), khác cách chia |
| Giai đoạn B1 (Claude Code) | z-score 0.042, Isolation Forest 0.019, HGB 0.586 (PR-AUC) | **Không trực tiếp**: khác nhãn (1 364 dương theo cửa sổ) và khác cách chia |

Nên để so **công bằng** với Hybrid của bạn cần chạy H3 bên dưới: cùng ma trận 330 452 × 90, cùng mốc cắt 2010-10-01, cùng chỉ số.

## 1. "Tái hiện bài" có hai mức, đừng lẫn

1. **Đúng như bài** (mỗi dòng log là một mẫu, đầu vào one-hot activity + thứ + giờ = 38 chiều, chấm F1 theo dòng). Giai đoạn A đã làm và không tái lập được. Không cần làm lại.
2. **Phương pháp của bài đặt trên dữ liệu của bạn** (H3): AE 100-50-50-50-100 và VAE latent 2, học trên user-day bình thường, điểm là lỗi tái tạo. Script `h3_pantelidis_userday.py` làm việc này.

Mức 2 trả lời câu hỏi: "chỉ một autoencoder theo kiểu bài, trên cùng đặc trưng, có đạt AUC-PR ngang Hybrid A (0.0696) không?" Số AE dạng bảng một lớp của bạn đã là 0.0708, nên bạn nên chuẩn bị tinh thần rằng kết quả có thể xấp xỉ nhau.

## 2. Chạy

Cần: Python có `torch`, `numpy`, `pandas`, `scikit-learn`, và `pyarrow` (để đọc parquet). Không cần GPU.

```bash
# Bước 1: xem tên cột của hai file (không huấn luyện gì)
python h3_pantelidis_userday.py --features features_userday.parquet --labels labels_userday.parquet --inspect

# Bước 2: chạy thử nhanh (2 seed, ít epoch) để chắc đường dẫn và tên cột đúng
python h3_pantelidis_userday.py --features features_userday.parquet --labels labels_userday.parquet \
    --out out_h3_smoke --seeds 2 --epochs 5

# Bước 3: chạy đủ (5 seed, AE và VAE)
python h3_pantelidis_userday.py --features features_userday.parquet --labels labels_userday.parquet \
    --out out_h3
```

Nếu script không tự nhận ra cột, truyền tay: `--label-col`, `--split-col`, `--scenario-col`, `--day-col`, `--user-col`.

## 3. Kiểm tra trước khi tin kết quả

Script in các dòng sau; đối chiếu với báo cáo của bạn:

- Số đặc trưng = 90.
- Train khoảng 189 237 dòng, test 141 140 dòng, 479 dương, 229 ngày test. Nếu `[split]` in ra khác, cách chia đang khác của bạn (cột split có thể đặt tên khác, hoặc 75 user-day của 6 insider nằm giữa hai kỳ chưa bị loại).
- Mức ngẫu nhiên AUC-PR in ra ≈ 0.0034.
- Tiền xử lý: mặc định script dùng log1p có dấu rồi chuẩn hoá bằng μ/σ của dòng bình thường kỳ train. **Cách này là giả định của tôi.** Nếu bạn có hàm tiền xử lý thật (`pipeline.py`) và file đã chuẩn hoá, hãy dùng ma trận đó và thêm `--no-log1p`. Hai cách khác nhau sẽ cho AUC-PR khác nhau.

## 4. Chạy mất bao lâu

Trong thử nghiệm của tôi (dữ liệu giả, 72 000 dòng, CPU 2 nhân), một lần huấn luyện 15 epoch mất khoảng 15–30 giây. Trên dữ liệu thật (khoảng 185 000 dòng, tối đa 100 epoch, dừng sớm) tôi ước tính vài phút mỗi lần, 10 lần chạy trong khoảng một giờ trên CPU. Đây là ước tính, **chưa đo trên dữ liệu thật**.

## 5. Điều script đảm bảo và điều không

- Epoch tốt nhất chọn bằng lỗi tái tạo trên 30 ngày cuối kỳ train (chỉ dòng bình thường). Test chỉ được chấm một lần cho mỗi (mô hình, seed) sau khi huấn luyện xong.
- Ngưỡng cảnh báo = phân vị 95% của điểm trên train bình thường (cùng quy tắc R² của bạn).
- Đầu ra: `h3_per_seed.csv` (mỗi seed một dòng), `h3_summary.csv` (trung bình, std), `h3_meta.json`. Cột AUC-PR và rec@k điền thẳng vào sheet "Kết quả 5 seed" (xlsx).
- **Đã kiểm tra:** chạy xuyên suốt trên dữ liệu giả, và hai hàm chỉ số đúng trên ví dụ có đáp án biết trước (rec@k tính theo `k × số ngày test`).
- **Chưa kiểm tra:** chạy trên dữ liệu thật của bạn. Tôi không có file đó trong môi trường này.

## 6. Khác với bài (ghi vào luận văn)

1. Đầu vào là vector user-day 90 chiều liên tục, không phải one-hot 38 chiều.
2. Đầu ra tuyến tính và mất mát MSE, vì đầu vào không nằm trong [0, 1] (bài dùng sigmoid/BCE).
3. Adam lr 1e-3, batch 256, dừng sớm; bài dùng 30 epoch cho AE và 1000 epoch cho VAE với Nadam.
4. Phân bổ tanh/ReLU giữa các lớp là lựa chọn của chúng tôi vì bài không nêu.
5. PyTorch thay Keras.

## 7. Lưu ý khi đọc bài

Trong đoạn bạn chụp có câu "80% of the samples (i.e., 769 images) were labelled as containing malicious activity and 430 as normal". Theo con số thì 769/(769+430) ≈ 64%, không phải 80%, và tỉ lệ lớp độc hại cao hơn rất nhiều so với 0.34% ở kỳ test của bạn. Tôi chỉ thấy đoạn trích nên không kết luận bài sai; nên hỏi lại cách bài chia dữ liệu trước khi đặt số 92–94% cạnh AUC-PR của bạn.

## 8. Đọc kết quả

Dùng quy tắc đã đặt trong xlsx: "hơn" khi AUC-PR cao hơn ở ít nhất 4/5 seed và tỉ số trung bình từ 1.10 trở lên. Nếu H3 và Hybrid A nằm trong khoảng 10% của nhau thì ghi là xấp xỉ.
