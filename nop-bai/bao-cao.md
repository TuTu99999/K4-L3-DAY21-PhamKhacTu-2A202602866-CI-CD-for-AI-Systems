# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phạm Khắc Tú |
| MSSV | 2A202602866 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/TuTu99999/K4-L3-DAY21-PhamKhacTu-2A202602866-CI-CD-for-AI-Systems |
| Ngày nộp | 08/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 có F1 `0.7149`, cao nhất và vượt ngưỡng 0.65. Lần 1 đạt accuracy cao nhất (`0.8780`) nhưng F1 chỉ `0.7109`, chứng tỏ accuracy không phản ánh đầy đủ lớp thiểu số. Lần 2 vừa giảm learning rate vừa giảm số cây nên thiếu năng lực học và không qua quality gate. Learning rate nhỏ thường cần nhiều cây hơn; cấu hình 200 cây, learning rate 0.1 cân bằng tốt nhất dù accuracy giảm nhẹ.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu Adult Income thuộc lớp trên 50K. Mô hình luôn đoán `thu_nhap_thap` vẫn đạt accuracy `0.752` dù không phát hiện mẫu dương nào, nên accuracy dễ gây hiểu nhầm. F1 của lớp dương cân bằng precision và recall, phản ánh cả độ chính xác lẫn khả năng tìm đủ người thu nhập cao. Tôi dùng `f1_score(y_eval, preds)` để đo trực tiếp lớp này. Không dùng `average="weighted"` vì lớp đa số sẽ kéo điểm lên; cũng không dùng `average="macro"` vì mục tiêu là hiệu quả riêng trên lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| GCP trả lỗi billing 403 | Tài khoản chưa kích hoạt billing. | Chuyển sang AWS theo đề. |
| Terminal không nhận AWS CLI | VS Code chưa cập nhật PATH. | Dùng `aws.exe` với profile `lab`. |
| S3 từ chối tạo bucket | IAM user thiếu `s3:CreateBucket`. | Cấp policy, tạo bucket và xác nhận `dvc push`. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Với 44.722 mẫu, F1 tăng `0.0205` và accuracy tăng `0.0080`. Mức tăng vừa phải phù hợp vì hai batch cùng phân phối; quan trọng hơn, commit chỉ đổi con trỏ DVC đã tự chạy đủ bốn job và triển khai lại API.

---

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: Chưa thực hiện.
- [x] Bonus 2: Quét ngưỡng 0,1–0,9; ngưỡng 0,3 đạt F1 `0.7537`.
- [x] Bonus 3: Lưu precision `0.7014`, recall `0.8145` và confusion matrix.
- [x] Bonus 4: Chỉ promote candidate khi F1 không giảm.
- [x] Bonus 5: Cảnh báo khi tỷ lệ lớp dương lệch quá 5 điểm phần trăm.
