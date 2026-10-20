# Báo cáo: dựng và tái hiện Deep SAD (Ruff et al.) trên CERT r4.2

Cập nhật 2026-10-20. Dự án: `D:\Code\paper\resources\resource_benmark`.

Báo cáo đi theo thứ tự thời gian, từ bản thiết kế tới kết quả benchmark. Mỗi kết quả kèm lệnh đã chạy và file đầu ra. Phụ lục A là danh sách lệnh để chạy lại từ đầu.

---

## 0. Kết quả trong một đoạn

Deep SVDD (Ruff et al., ICML 2018) và Deep SAD (Ruff et al., ICLR 2020) đã được dựng lại bằng PyTorch. Bản dựng đã được đối chiếu từng dòng với mã tham chiếu `deepsad_bn2.py` của bạn: phép toán khớp tới sai số làm tròn (≤ 1.6e-15). Sau đó hai mô hình được chạy trên CERT r4.2 theo **đúng giao thức trong báo cáo Hybrid của bạn**: mốc cắt 2010-10-01, nhãn user-day theo sự kiện, 141 140 dòng test với 479 dương, chỉ số rec@k và S1–S3@25. Kết quả trung bình 5 seed:
- **Deep SAD**, η = 1, 460 nhãn thật: AUC-PR **0,912 ± 0,012**, ROC-AUC 0,9986, rec@25 0,9925.
- **Deep SVDD**, không dùng nhãn: AUC-PR **0,092 ± 0,027**, ROC-AUC 0,900, rec@25 0,585.

Cách chia và nhãn khớp từng con số với báo cáo, nhưng **bộ đặc trưng khác**: ở đây là 88 cột G+D của dự án, không phải 90 cột của partner. Vì vậy các con số này không so thẳng được với Hybrid (0,0696 và 0,8585).

---

## 1. Dòng thời gian

| # | Thời điểm | Việc | Ai chạy, ở đâu |
|---|---|---|---|
| 0 | trước 2026-10-17 | Tái hiện Deep SAD trên 6 bộ ODDS bằng `deepsad_bn2.py`, ra bảng `Downloads\results.md` | **Không chạy trên máy này.** File đọc dữ liệu ở `/home/claude/deepsad/...`; mình chỉ đọc bảng kết quả |
| 1 | 2026-10-17 | Viết thiết kế B2 (Deep SVDD, Deep SAD, Temporal AE) vào `docs/phaseB_spec.md`, mục B2-0, trước mọi lần chạy | mình |
| 2 | 2026-10-19 | Bạn duyệt B2-0 kèm 6 điều chỉnh (đẩy tâm theo Ruff, ghi rõ lệch lịch lr, test j/k, chạy thử 1 fold rồi dừng) | bạn |
| 3 | 2026-10-19 | Viết code `src/deep_b2.py` và `src/run_b2.py`, test `tests/test_phaseB2.py` | mình |
| 4 | 2026-10-19 | Chạy test f–k; chạy thử 1 fold PB-U r4.2 (fit + val, không chấm test) | mình, GPU |
| 5 | 2026-10-19 | Nhận `deepsad_bn2.py`; đối chiếu từng dòng; chạy giao thức của file trên fold chạy thử | mình, GPU |
| 6 | 2026-10-19/20 | Benchmark r4.2 theo giao thức báo cáo Hybrid: Deep SVDD, Deep SAD (kèm Pantelidis AE/VAE), 5 seed | mình, GPU |

Lưới đầy đủ B2 (10 fold PB-U, PB-T, r6.2) **chưa chạy**, vẫn đang chờ bạn xác nhận.

---

## 2. Bước 0: tái hiện trên ODDS (không chạy trên máy này)

Bảng `C:\Users\binh5\Downloads\results.md`: AUC (%), 10 seed, γ_l = 0,01, so với Bảng 2 của Ruff et al. 2020.

| Bộ dữ liệu | Bài | BN (bản chính) | Trong ±1 std của bài? |
|---|---|---|---|
| arrhythmia | 75.9 ± 8.7 | 77.6 ± 4.8 | có |
| cardio | 95.0 ± 1.6 | 90.1 ± 4.5 | không (−4.9) |
| satellite | 91.5 ± 1.1 | 91.4 ± 1.3 | có |
| satimage-2 | 99.9 ± 0.1 | 99.3 ± 0.7 | không (−0.6) |
| shuttle | 98.4 ± 0.9 | 99.96 ± 0.03 | không (+1.6) |
| thyroid | 98.6 ± 0.9 | 99.4 ± 0.3 | có |

- Bước này cho thấy cấu hình trong `deepsad_bn2.py` (BN không affine, LeakyReLU 0.1, lịch lr hai pha, batch 200, wd 1e-6) tái hiện được bài ở 3/6 bộ, trong phạm vi 1 độ lệch chuẩn.
- Lệnh tương ứng trong file là `python deepsad_bn2.py <dataset> 10`.
- **Mình không chạy lại được** trên máy này, vì máy không có các file `*.npz` của ODDS. Tải về phải ra ngoài sandbox, nên chưa làm.

---

## 3. Dữ liệu (dùng chung cho các bước 4–6)

| Bước | Lệnh | Đầu ra |
|---|---|---|
| Danh sách file HF và lfs.oid | `python src/hf_fetch.py manifest` | `results/hf_manifest.json` (commit `4107ac90…`) |
| Kiểm bản local so với HF | `python src/hf_fetch.py verify-local D:\Code\paper\data\cert_r4.2 cert_r4.2 "*"` | `results/hash_log.jsonl` (sha256 khớp lfs.oid) |
| Tải nhãn | `python src/hf_fetch.py get "cmu_kilthub/answers/*"` | `answers.tar.bz2`, sha256 đã kiểm |
| Nhãn từ `answers.tar.bz2` | `python src/labels.py 4.2` | `data/labels/insiders_r4.2.csv`, `malicious_events_r4.2.csv` (70 insider) |
| Bảng đếm log | `python src/build_counts.py 4.2` | `data/counts/r4.2/*.parquet` |
| Đặc trưng và nhãn user-day | `python src/userday.py 4.2` | `data/userday/r4.2.parquet` (330 452 × 95), `data/userday/labels_r4.2.parquet` |

**Nguồn dữ liệu:**
- Bản `cert_r4.2` trên máy trùng sha256 với lfs.oid trên HF ở commit đã ghim.
- Ngày 2026-10-20, cả 8 file ở nhánh `main` có kích thước trùng từng byte với bản ghim. Mình chưa lấy được oid ở `main` vì kết nối tới HF liên tục bị reset.

**Đặc trưng dùng cho Deep SVDD/SAD:** bộ G+D gồm 88 cột, trong đó 2 cột hằng bị bỏ.
- Nhóm G là đếm theo hoạt động × khung giờ, ngoài giờ, phiên, PC, USB, tệp, email, http.
- Nhóm D là độ lệch so với lịch sử **trước** ngày đang xét của chính người đó, nên không rò rỉ dù cắt ở mốc nào.
- Không dùng nhóm từ khoá URL (I), OCEAN hay LDAP.

---

## 4. Bước 1–3: thiết kế và code

### 4.1 Mô hình (spec B2-0.d, `src/deep_b2.py`)

- **Encoder φ:** `Linear(d, 64, bias=False) → BN(64, eps 1e-4, affine=False) → LeakyReLU(0.1) → Linear(64, rep, bias=False)`. Không có bias ở bất kỳ lớp nào, vì bias hoặc BN affine cho phép nghiệm sụp về tâm.
- **Decoder** (chỉ dùng khi pretrain AE): đối xứng với encoder, đầu ra tuyến tính.
- **Tâm c:** trung bình φ ở chế độ eval sau pretrain AE. Mọi |c_j| < 0,1 được đặt thành 0,1 · sign(c_j); c_j = 0 thành +0,1 (dòng 92–93 của mã tham chiếu).
- **Loss Deep SAD:** hàng không nhãn dùng ‖φ − c‖²; hàng có nhãn bất thường dùng η · (‖φ − c‖² + 1e-6)⁻¹; lấy trung bình theo batch. Deep SVDD là cùng loss này với tập nhãn rỗng.
- **Điểm bất thường:** ‖φ(x) − c‖² (cùng thứ hạng với ‖φ − c‖ của mã tham chiếu).

### 4.2 Cách cài đặt

- **`StackMLP` và `StackAdam`:** K mô hình độc lập chạy trong một tiến trình, trọng số dạng (K, vào, ra) nhân bằng `bmm`. BN và Adam được viết tay cho từng mô hình. Mỗi mô hình có tập fit, seed và bộ sinh số ngẫu nhiên riêng; mô hình xong trước thì được đóng băng. Nhờ vậy 5–10 mô hình chạy gần như với giá của một mô hình.
- **`LabelGate`:** nhãn nằm sau một cổng.
  - Deep SAD chỉ đọc được nhãn dương của tập fit.
  - Nhãn val/test chỉ mở theo thứ tự train → val → test, và test được chấm đúng một lần.
- **Tất định:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms(True)`, cuDNN deterministic, tắt TF32.

### 4.3 Kiểm thử: `python -m pytest tests/test_phaseB2.py` (32/32 đạt cùng các test B1)

| Test | Nội dung | Kết quả |
|---|---|---|
| (g) | Bản xếp chồng so với bản `nn.Linear` + `nn.BatchNorm1d` + `torch.optim.Adam` chạy riêng từng mô hình (CPU, float64). Gồm cả việc đóng băng mô hình xong trước, đảo thứ tự trong chồng, chạy một mình, và đổi seed/dữ liệu của mô hình khác | sai khác ≈ 1e-16 (trọng số dịch ≈ 1e-2) |
| (g), kiểm tra test | Cố tình cài 3 lỗi (momentum BN, Adam lệch 1e-9, bỏ đóng băng) để xem test có bắt được không (script tạm `mutate_g.py` trong scratchpad) | cả 3 lần test đều báo trượt; mã đã khôi phục, md5 khớp |
| (i) | Đẩy tâm và loss Deep SAD trên batch dựng tay | đạt |
| (j) | Nhãn dương của Deep SAD ⊂ tập fit; đọc val/test khi cổng còn đóng thì báo lỗi; test chỉ chấm một lần | đạt |
| (f) | Chạy hai lần trên GPU phải giống nhau từng bit | đạt, sai khác = 0 (`results/phaseB/b2_determinism.json`) |
| (h), (k) | Cửa sổ của Temporal AE (không thuộc Deep SAD) | đạt |

---

## 5. Bước 4: chạy thử 1 fold (fit + val, không chấm test)

**Lệnh:** `python src/run_b2.py smoke`. **Đầu ra:** `results/phaseB/b2_smoke.json`, `logs/phaseB2_smoke.log`.

Cấu hình: r4.2 PB-U lần chạy 0; 211 334 hàng fit, 52 243 hàng val; rep 16; lr cố định 1e-4 × 150 epoch, batch 256 (theo spec B2-0).

| | Loss (epoch 1 → 150) | NaN | Trung vị khoảng cách trên val (khởi tạo → sau huấn luyện) | std của φ |
|---|---|---|---|---|
| AE pretrain | 0,155 → 0,0173 | không | — | — |
| Deep SVDD | 1,022 → 0,00022 | không | 2,95 → 2,6e-5 | 0,0021 |
| Deep SAD, η = 1 (829 nhãn) | 1,028 → 0,0099 | không | 2,95 → 9,0e-4 | 0,034 |

**Nhận xét:**
- Deep SVDD **co rất mạnh về c** (khoảng 10⁵ lần) nhưng không sụp hẳn: điểm vẫn tách được và không chiều nào có std < 1e-3.
- Deep SAD đẩy các hàng có nhãn ra xa: khoảng cách trung bình trên fit là 4,05 với hàng có nhãn và 0,0036 với hàng không nhãn.

---

## 6. Bước 5: đối chiếu với mã tham chiếu `deepsad_bn2.py`

File được chép nguyên vào `docs/reference/deepsad_bn2.py` (sha256 `de1851be…ee035c`). Bảng đầy đủ nằm ở spec mục B2-0.f.

### 6.1 Đối chiếu bằng số

**Lệnh:** `python src/refcheck_b2.py equiv`. **Đầu ra:** `results/phaseB/b2_refcheck_equiv.json`.

| Điểm | Sai khác đo được |
|---|---|
| Đẩy tâm (dòng 92–93) | 0 |
| Lớp và BN (`mlp`, dòng 36–43) so với `StackMLP`, cùng trọng số | forward ở chế độ train 1,1e-15; eval 2,8e-16; running stats ≤ 2,2e-16 |
| `torch.optim.Adam(weight_decay)` so với `StackAdam`, 20 bước | 1,6e-15 |
| Loss Deep SAD (dòng 98), η = 0,5 / 1 / 2 | 0 |
| `one_seed()` của file chạy trọn trên tập tổng hợp (chế độ quick) | AUC 0,97, chạy được |

**Lỗi tìm ra và đã sửa:** `push_center` tạo hằng 0,1 ở float32, nên khi chạy float64 lệch 1,5e-9. Các lần chạy thật dùng float32 nên số liệu không đổi.

### 6.2 Điểm khác giữa spec B2 và mã tham chiếu

| # | Mã tham chiếu | Spec B2 |
|---|---|---|
| 1 | Adam 1e-4 × 50, rồi Adam **mới** 1e-5 × 100 | 1e-4 cố định × 150 (đã ghi là lệch) |
| 2 | batch 200, giữ batch lẻ cuối | batch 256, bỏ batch lẻ |
| 3 | Loss AE là tổng theo đặc trưng | trung bình (chênh đúng 88 lần) |
| 4 | AE pretrain và c chỉ trên hàng không nhãn | mọi hàng fit; Deep SAD dùng chung AE với Deep SVDD |
| 5 | 3 lớp, chọn theo từng bộ ODDS | 2 lớp d-64-rep |
| 6 | Chuẩn hoá bằng StandardScaler | chuẩn hoá rồi signed log1p |

Trong chính file tham chiếu cũng có một chỗ mâu thuẫn: docstring ghi "batch-norm omitted", nhưng code có BN.

### 6.3 Chạy giao thức của mã tham chiếu trên fold chạy thử

**Lệnh:** `python src/refcheck_b2.py proto`. **Đầu ra:** `results/phaseB/b2_refcheck_proto.json`, `logs/phaseB2_refcheck.log`.

Áp các điểm 1–4 của mã tham chiếu; giữ kiến trúc và tiền xử lý của spec.

| | Mã tham chiếu: SVDD | Spec: SVDD | Mã tham chiếu: SAD | Spec: SAD |
|---|---|---|---|---|
| Loss cuối | 0,00026 | 0,00022 | 0,013 | 0,0099 |
| Trung vị khoảng cách trên val | 2,2e-5 | 2,6e-5 | 1,6e-3 | 9,0e-4 |
| std của φ | 0,0026 | 0,0021 | 0,028 | 0,034 |
| Khoảng cách trên fit, có nhãn / không nhãn | — | — | 2,34 / 0,0056 | 4,05 / 0,0036 |

→ Chạy theo mã tham chiếu cũng thấy hiện tượng co của Deep SVDD **y như vậy**, nên nó không do các điểm lệch 1–4 gây ra. Deep SAD cho cùng một kiểu kết quả, các con số chênh nhau trong khoảng 2 lần.

---

## 7. Bước 6: benchmark r4.2 theo giao thức báo cáo Hybrid

**Lệnh:**
```
python src/bench_r42.py smoke              # 2 seed, ít epoch, không đọc nhãn test
python src/bench_r42.py run PAE,RUFF       # Pantelidis AE + Ruff Deep SVDD/SAD, 5 seed (49 phút)
python src/bench_r42.py run PVAE           # Pantelidis VAE, 5 seed, 1000 epoch (3,5 giờ), chạy song song
python src/bench_r42.py report             # đọc nhãn test MỘT lần, tính chỉ số
```
**Đầu ra:**
- `results/bench_r42/`: `summary.csv`, `per_seed.csv`, `scores_*.npz`, `meta_*.json`, `BAO_CAO_bench_r42.md`;
- log ở `logs/bench_r42.log`.

### 7.1 Giao thức: khớp từng con số với `BAO_CAO_TONG_KET_TOAN_BO.md` mục 3.6

| | Ở đây | Báo cáo Hybrid |
|---|---|---|
| Train (≤ 2010-09-30) | 189 237 dòng, 460 dương (45 / 404 / 11) | như vậy |
| Test (≥ 2010-10-01) | 141 140 dòng, 479 dương (40 / 431 / 8), 229 ngày | như vậy |
| Insider vắt ngang | gán về phía có nhiều ngày tấn công hơn (hoà thì về train); xoá 75 user-day, trong đó 27 dương | như vậy |
| Nhãn | `y_event` (ngày có ít nhất một sự kiện độc hại) | như vậy (966) |
| rec@k, S1–S3@25 | recall trong k × 229 dòng có điểm cao nhất | như vậy |

### 7.2 Cấu hình Ruff trong benchmark

Cấu hình bám mã tham chiếu và được cố định trước khi đọc nhãn test:
- **Mạng:** MLP không bias d-64-32, BN không affine, LeakyReLU 0,1.
- **Huấn luyện:** AE pretrain (loss tổng theo đặc trưng), rồi Deep SVDD/SAD. Mỗi pha dùng Adam 1e-4 × 50, sau đó Adam mới 1e-5 × 100; batch 200; weight decay 1e-6.
- **Dữ liệu huấn luyện:** AE và c học trên các dòng bình thường của kỳ train. Deep SAD gắn nhãn đủ 460 user-day dương của kỳ train, η = 1.
- **Tiền xử lý:** signed log1p, rồi z-score với μ, σ của các dòng bình thường kỳ train (theo báo cáo, mục 3.5).
- **Seed:** 0–4; không tinh chỉnh gì.

### 7.3 Kết quả (trung bình 5 seed; độ lệch chuẩn nằm trong `summary.csv`)

| cấu hình | ROC-AUC | AUC-PR | rec@5 | rec@10 | rec@25 | rec@50 | S1@25 | S2@25 | S3@25 |
|---|---|---|---|---|---|---|---|---|---|
| Ruff Deep SVDD, không nhãn bất thường | 0,8999 | 0,0922 | 0,2718 | 0,3958 | 0,5854 | 0,7090 | 0,850 | 0,5564 | 0,825 |
| **Ruff Deep SAD, η=1, 460 mẫu thật** | **0,9986** | **0,9124** | 0,9653 | 0,9816 | 0,9925 | 0,9975 | 0,965 | 0,9968 | 0,900 |
| *(cùng lần chạy)* Pantelidis AE | 0,9791 | 0,4602 | 0,6413 | 0,7144 | 0,8225 | 0,9040 | 0,965 | 0,8060 | 1,000 |
| *(cùng lần chạy)* Pantelidis VAE | 0,9716 | 0,1462 | 0,3624 | 0,5599 | 0,7775 | 0,9019 | 0,920 | 0,7601 | 1,000 |

AUC-PR theo từng seed:
- Deep SAD: 0,916 / 0,896 / 0,904 / 0,922 / 0,924.
- Deep SVDD: 0,100 / 0,084 / 0,133 / 0,059 / 0,084.

Loss của Ruff (trung bình 5 seed):
- AE: 52,1 → 7,98 ở epoch 50 → 7,46 ở epoch 150.
- Deep SVDD: 1,33 → 0,00079.
- Deep SAD: 1,32 → 0,0066.

Không có NaN.

### 7.4 Đặt cạnh báo cáo Hybrid (chỉ để tham khảo)

| | AUC-PR |
|---|---|
| Hybrid A (báo cáo, 0 nhãn, 1 seed) | 0,0696 |
| Ruff Deep SVDD (ở đây, 5 seed) | 0,0922 ± 0,0274 |
| Hybrid + Deep SAD, 460 mẫu thật (báo cáo, 1 seed) | 0,8585 |
| Ruff Deep SAD, 460 mẫu thật (ở đây, 5 seed) | 0,9124 ± 0,0122 |
| Random Forest (báo cáo) | 0,9302 |

**Không phải phép so công bằng:**
- Hai bên dùng cùng nhãn, cùng cách chia và cùng chỉ số, nhưng khác bộ đặc trưng (88 cột G+D so với 90 cột của partner).
- Hai bên khác đơn vị đầu vào: ở đây là từng user-day; Hybrid dùng cửa sổ 7 ngày.
- Quy tắc "hơn" (ít nhất 4/5 seed và tỉ số ≥ 1,10) không áp dụng được, vì phía báo cáo chỉ có một seed.

---

## 8. Toàn bộ điểm lệch khỏi bài gốc (ghi vào luận văn)

1. **Dữ liệu và đơn vị:** CERT user-day dạng bảng, không phải ảnh hay ODDS. Tiền xử lý dùng signed log1p cộng z-score, không phải min-max hay StandardScaler.
2. **Kiến trúc:** 2 lớp d-64-rep; bài dùng 3 lớp và chọn theo từng bộ dữ liệu. Decoder có đầu ra tuyến tính, không có sigmoid.
3. **γ_l:** bài dùng 1% nhãn. Ở đây gắn nhãn **mọi** dòng dương của kỳ train: 460 / 189 237 ≈ 0,24% ở benchmark, 0,39% ở chạy thử.
4. **Spec B2** (lần chạy thử, và lưới đầy đủ nếu được duyệt): lr cố định 1e-4 × 150, batch 256, loss AE trung bình, AE và c học trên mọi hàng fit. **Benchmark (bước 6)** thì theo mã tham chiếu: lịch hai pha, batch 200, loss tổng, AE và c chỉ trên hàng bình thường.
5. **Batch lẻ cuối:** bản xếp chồng bỏ batch lẻ cuối, còn mã tham chiếu giữ lại.

---

## 9. Lỗi và sự cố đã gặp

| Sự cố | Xử lý |
|---|---|
| File `deepsad_bn2.py` không đi kèm tin nhắn duyệt B2-0 | Lúc đầu đối chiếu theo các giá trị trong tin nhắn; nhận file sau thì đối chiếu lại từng dòng (mục 6) |
| `push_center` lệch 1,5e-9 ở float64 | Đã sửa; float32 không đổi; test chạy lại vẫn đạt |
| Lần chạy nền đầu tiên lỗi `FileNotFoundError` (đường dẫn python tương đối) | Chạy lại với đường dẫn tuyệt đối |
| Kết nối HF bị reset liên tục | Chỉ lấy được danh sách file và kích thước, không lấy được oid và commit của `main` |
| Không có file đặc trưng 90 cột của partner | Dùng bộ G+D của dự án; đã ghi rõ ở mọi bảng |

---

## 10. Việc còn lại

- **Lưới đầy đủ B2:** 10 fold PB-U và PB-T, r4.2 rồi r6.2, ước khoảng 32–35 giờ GPU nếu chạy lần lượt. Đang chờ bạn xác nhận.
- **So công bằng với Hybrid:** cần `features_userday.parquet` và `labels_userday.parquet` (90 cột) của partner, rồi chạy lại `bench_r42.py` trên đúng ma trận đó.
- **ODDS:** chạy lại `deepsad_bn2.py` trên máy này nếu bạn đồng ý tải dữ liệu ODDS. Khoảng cách −4,9 điểm ở cardio vẫn chưa giải thích được.

---

## Phụ lục A: chạy lại từ đầu

Chạy trong thư mục dự án, bằng `.venv\Scripts\python.exe`. Biến môi trường `HF_TOKEN` phải được đặt sẵn; không ghi token vào file.

```
python src/hf_fetch.py manifest
python src/hf_fetch.py verify-local D:\Code\paper\data\cert_r4.2 cert_r4.2 "*"    # hoặc: get "cert_r4.2/*"
python src/hf_fetch.py get "cmu_kilthub/answers/*"
python src/labels.py 4.2
python src/build_counts.py 4.2
python src/userday.py 4.2
python -m pytest tests/test_phaseB2.py           # test f-k (f cần GPU)
python src/run_b2.py smoke                       # chạy thử 1 fold, khoảng 40 phút
python src/refcheck_b2.py equiv                  # đối chiếu bằng số với deepsad_bn2.py
python src/refcheck_b2.py proto                  # giao thức tham chiếu trên fold chạy thử, khoảng 60 phút
python src/bench_r42.py run PAE,RUFF             # khoảng 49 phút
python src/bench_r42.py run PVAE                 # khoảng 3,5 giờ (có thể chạy song song với lệnh trên)
python src/bench_r42.py report
```

Các bước trên 5 phút đã được chạy nền kèm bộ giám sát RAM (`memguard.py` trong scratchpad, không thuộc dự án).

## Phụ lục B: file liên quan

| File | Vai trò |
|---|---|
| `docs/phaseB_spec.md` (§8, B2-0.c–f) | Thiết kế, các điều chỉnh khi duyệt, bảng đối chiếu tham chiếu |
| `docs/reference/deepsad_bn2.py`, `README.md` | Mã tham chiếu của bạn (giữ nguyên) |
| `src/deep_b2.py` | StackMLP, StackAdam, loss, đẩy tâm, LabelGate, Temporal AE |
| `src/run_b2.py` | Chạy B2 theo spec: smoke / ae / deep / tae / score |
| `src/refcheck_b2.py` | Đối chiếu với mã tham chiếu: equiv / proto |
| `src/bench_r42.py` | Benchmark theo giao thức báo cáo Hybrid |
| `tests/test_phaseB2.py` | Test f–k |
| `results/phaseB/b2_smoke.json`, `b2_determinism.json`, `b2_refcheck_equiv.json`, `b2_refcheck_proto.json` | Kết quả các bước 4–5 |
| `results/bench_r42/` | Kết quả bước 6 |
