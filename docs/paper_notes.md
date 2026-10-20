# Ghi chú đọc bài — Pantelidis et al., IEEE 2021 (6 trang)

File: `docs/Insider-Threat-Detection-using-Deep-Autoencoder-and-Variational-Autoencoder-Neural-Networks.pdf`

## Bài nêu rõ
| Mục | Nội dung trong bài | Trang |
|---|---|---|
| Dữ liệu | CERT r4.2, 1000 user, logon/device/http/email/file | 3 |
| Lấy mẫu | "sample files of 5000 entries per column for each of the CSV files" | 4 |
| Đặc trưng | day 0–6, time 1–24, activity 1–7 (Logon, Logoff, Connect, Disconnect, email, file, http), rồi one-hot | 4, Bảng II |
| Nhãn | cột `insider` = 1 nếu user nằm trong danh sách insider "according to the scenarios" | 4 |
| Chia | 75% train / 25% test | 4 |
| AE | input 50 → 100 → 50 → bottleneck 50 → decoder đối xứng; activation "(tanh, relu)" | 4 |
| AE train | 30 epoch, batch 256 | 5 |
| VAE | encoder ra μ, σ; latent 2; Nadam; 1000 epoch, batch 128 | 4–5 |
| LR | "optimal values of these parameters identified using LRFinder [29]", không ghi giá trị | 5 |
| Framework | Keras + TensorFlow | 4 |
| Kết quả | AE P/R/A/F1 = 90/95/95/92; VAE 92/96/96/94 (Bảng III, Hình 3) | 5 |

## Mâu thuẫn / thiếu (phải tự chọn, ghi vào REPORT)
1. **Số insider:** bài ghi "30 malicious employees" (tr. 3), nhưng `answers` của r4.2 có **70** insider (30/30/10 cho kịch bản 1/2/3). Bài không nói danh sách 30 user lấy từ đâu.
2. **Kích thước input 50:** one-hot của day (7) + time (24) + activity (7) = **38**, không phải 50. Bài không giải thích 12 chiều còn lại.
3. **Cơ chế phân loại:** bài không nói anomaly score/ngưỡng là gì. Bài còn viết "the decoder can be dropped… keep only the trained encoder", điều này mâu thuẫn với việc chấm điểm bằng lỗi tái tạo.
4. **Train "with labelled data":** không rõ AE/VAE học trên mọi dòng train, chỉ dòng bình thường, hay theo kiểu có giám sát.
5. **"5000 entries per column" và "single CSV file with entries from 1000 rows":** hai con số không khớp nhau, cũng không nói lấy 5000 dòng đầu hay ngẫu nhiên.
6. **VAE:** không có kích thước lớp ẩn, hàm loss (MSE hay BCE, trọng số KL), activation.
7. **AE:** không ghi optimizer, loss, learning rate.
8. **Công thức metric:** định nghĩa bằng lời của Recall/FP/FN bị sai (Recall mô tả giống specificity; FN được định nghĩa là "normal activities incorrectly classified"); công thức F1 thiếu hệ số 2. Mình dùng công thức chuẩn: P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R).
9. Chi tiết nhỏ: bài nói dữ liệu từ 01.01.2010 đến 31.05.2011, nhưng logon r4.2 thật sự kết thúc ngày 2011-05-17; Bảng I gọi là "seven log files" (gồm psychometric và ldap); trích dẫn [21] cho dataset SEI thực ra là bài LSTM.
