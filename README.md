# Tóm tắt văn bản tiếng Anh

## Phương pháp

Đồ án tóm tắt trích xuất bằng xếp hạng trên đồ thị. Mỗi câu là một đỉnh. Trọng số cạnh là độ tương đồng cosine giữa hai vector tần suất từ do scikit-learn tạo. NLTK tách câu tiếng Anh.

PageRank được cài bằng Python với hệ số suy giảm 0,65. Thuật toán bỏ cạnh tự nối, chuẩn hóa trọng số đi ra và phân phối đều phần điểm của đỉnh không có cạnh đi ra. Việc cập nhật tiếp tục đến khi tổng độ thay đổi điểm nhỏ hơn 10⁻⁸.

Bước chọn câu dùng Maximal Marginal Relevance (MMR) để giảm trùng lặp. Ở mỗi lượt, điểm chọn bằng điểm PageRank chia cho điểm PageRank lớn nhất, trừ hai lần độ tương đồng lớn nhất với các câu đã chọn. Các câu được ghép theo thứ tự gốc. Cấu hình được chọn khi kiểm tra ví dụ Gettysburg trên slide. Bản tham chiếu chỉ dùng để kiểm tra kết quả sau khi sinh tóm tắt.

## Cài đặt và chạy

```powershell
python -m pip install -r requirements.txt
python -m nltk.downloader punkt_tab
python main.py
```

Chỉ cần cài NLTK và scikit-learn. Các lệnh dùng được trong môi trường Python 3.10 trở lên, bao gồm Anaconda.

Với văn bản có N câu và `ratio` trong khoảng `(0, 1]`, chương trình chọn `K = max(1, round(N * ratio))` câu. Văn bản rỗng cho bản tóm tắt rỗng.

Demo Gettysburg dùng `ratio=0.3` và sinh các câu 1, 4, 8. Dòng `Slide reference: MATCH` xác nhận bản sinh trùng với phần tô xanh trên slide.

## Đánh giá ROUGE

```powershell
python evaluate.py data/gettysburg.jsonl --ratio 0.3
python evaluate.py data/demo.jsonl --ratio 0.3
```

Tệp UTF-8 JSONL gồm một tài liệu mỗi dòng, với `text` là văn bản gốc và `reference` là bản tóm tắt tham chiếu. Lead-K lấy K câu đầu với cùng ngân sách số câu. Script tính trung bình ROUGE-1, ROUGE-2 và ROUGE-L F1 theo tài liệu, có stemming tiếng Anh. Trên ví dụ Gettysburg, phương pháp PageRank và MMR đạt 1,0000 ở cả ba chỉ số.

### Kết quả XSum của cấu hình trước

Bảng dưới đây lưu kết quả lịch sử của cấu hình TF-IDF trước khi đổi thuật toán chọn câu. Điểm là trung bình F1 theo tài liệu trên thang 0–100, tính trên 11.333 bài ở tỉ lệ 0,3. Bản ghi `39563665` có văn bản rỗng đã được loại. Cấu hình PageRank và MMR hiện tại mới được kiểm tra trên các ví dụ cục bộ. [Bài báo XSum](https://aclanthology.org/D18-1206/)

| Phương pháp | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| TextRank | 15,32 | **2,86** | 10,38 |
| Lead-K | **16,48** | 2,63 | **10,72** |
