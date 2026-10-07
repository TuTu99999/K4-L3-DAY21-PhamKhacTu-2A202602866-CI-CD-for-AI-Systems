# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | Phạm Khắc Tú |
| MSSV | 2A202602866 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/TuTu99999/K4-L3-DAY21-PhamKhacTu-2A202602866-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 được chọn vì đạt `f1_score=0.7149`, cao nhất trong ba lần và vượt ngưỡng chất lượng 0.65. Lần chạy 1 có accuracy cao nhất (`0.8780`) nhưng F1 thấp hơn (`0.7109`), cho thấy accuracy không phản ánh đầy đủ khả năng nhận diện lớp thu nhập cao vốn chiếm tỷ lệ nhỏ. Lần chạy 2 dùng ít cây hơn, cây nông hơn và learning rate thấp nên chỉ đạt F1 `0.6051`; mô hình chưa học đủ để vượt quality gate. Kết quả thể hiện quan hệ đánh đổi giữa `n_estimators` và `learning_rate`: learning rate nhỏ thường cần nhiều vòng boosting hơn, trong khi cấu hình lần 2 vừa giảm learning rate vừa giảm số cây. Tăng lên 200 cây với learning rate 0.1 cải thiện F1, dù accuracy giảm nhẹ so với lần 1.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

<!-- Khoảng 120 - 150 từ. -->

Tập Adult Income bị mất cân bằng: chỉ khoảng 24,8% mẫu thuộc lớp thu nhập trên 50K, còn 75,2% thuộc lớp thu nhập thấp. Vì vậy, một mô hình luôn dự đoán `thu_nhap_thap` vẫn đạt accuracy khoảng 0,752 dù không phát hiện được bất kỳ mẫu dương nào. Accuracy cao trong trường hợp này dễ tạo cảm giác sai rằng mô hình hoạt động tốt. F1-score của lớp dương cân bằng giữa precision và recall, nên phản ánh đồng thời mức độ dự đoán đúng thu nhập cao và khả năng tìm đủ các trường hợp thu nhập cao. Trong mã nguồn, tôi dùng `f1_score(y_eval, preds)` để đánh giá trực tiếp lớp dương. Tôi không dùng `average="weighted"` vì lớp đa số sẽ kéo kết quả lên, và cũng không dùng `average="macro"` vì yêu cầu chính là theo dõi riêng hiệu quả trên lớp thiểu số quan trọng.

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Không tạo được bucket trên GCP | Tài khoản chưa kích hoạt được billing nên Cloud Storage trả về lỗi 403. | Chuyển sang AWS theo lựa chọn cloud provider mà đề bài cho phép. |
| AWS CLI không được nhận diện trong terminal | Terminal VS Code được mở trước khi AWS CLI cập nhật PATH. | Gọi trực tiếp `aws.exe` theo đường dẫn cài đặt và dùng profile riêng `lab`. |
| IAM user bị từ chối khi tạo S3 bucket | User mới chưa có policy cho hành động `s3:CreateBucket`. | Gắn quyền S3 cho user, tạo bucket riêng và kiểm tra `dvc push` đồng bộ đủ ba object. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
