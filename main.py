# json chuyển nội dung JSON trong tệp mẫu thành dữ liệu Python.
import json
# Path giúp ghép và đọc đường dẫn trên các hệ điều hành.
from pathlib import Path

# Nhập hàm điều phối quá trình tóm tắt.
from src.summarizer import summarize_text


# Hàm chính chạy ví dụ Gettysburg và kiểm tra kết quả.
def main():
    # __file__ là đường dẫn của main.py. Lấy thư mục chứa tệp để tìm thư mục data.
    sample_path = Path(__file__).resolve().parent / "data" / "gettysburg.jsonl"
    # Đọc UTF-8 để giữ dấu câu Unicode. json.loads trả về dict có text và reference.
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    # Chỉ truyền văn bản gốc vào bộ tóm tắt. Tỉ lệ 0,3 cho 3 câu trên mẫu 9 câu.
    summary = summarize_text(sample["text"], ratio=0.3)

    # Mỗi phần tử là cặp tiêu đề và nội dung. Vòng lặp tháo cặp thành hai biến.
    for title, content in (
        ("ORIGINAL TEXT", sample["text"]),
        ("GENERATED SUMMARY (PAGERANK + MMR)", summary),
    ):
        # In đường phân cách bằng cách lặp ký tự = 60 lần.
        print("=" * 60)
        # In tiêu đề của phần hiện tại.
        print(title)
        print("=" * 60)
        # In văn bản gốc hoặc bản tóm tắt của phần hiện tại.
        print(content)
        # Thêm một dòng trống giữa các phần.
        print()

    # Chỉ sau khi có bản sinh mới đối chiếu nguyên chuỗi với đáp án lưu trong tệp.
    matches = summary == sample["reference"]
    # Biểu thức điều kiện trong f-string chọn nhãn theo giá trị True hoặc False.
    print(f"Slide reference: {'MATCH' if matches else 'MISMATCH'}")
    # Quy ước mã thoát 0 là khớp, 1 là lệch đáp án.
    return 0 if matches else 1


# Điều kiện đúng khi chạy python main.py. SystemExit gửi mã thoát về terminal.
if __name__ == "__main__":
    raise SystemExit(main())
