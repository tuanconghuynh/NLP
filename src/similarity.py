# CountVectorizer biểu diễn câu bằng số lần xuất hiện của từng từ.
from sklearn.feature_extraction.text import CountVectorizer
# Cosine đo độ giống về hướng giữa hai vector tần suất.
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(sentences):
    """Tính ma trận tương đồng giữa các câu."""

    # Dùng cấu hình mặc định. Bộ vector hóa chuyển về chữ thường và giữ token ít nhất hai ký tự.
    vectorizer = CountVectorizer()

    # fit tạo từ vựng chung. transform tạo ma trận thưa với mỗi hàng là một câu.
    try:
        counts = vectorizer.fit_transform(sentences)
    # Trường hợp đặc biệt xảy ra khi mọi câu đều thiếu token hợp lệ.
    except ValueError as error:
        # Chỉ dùng ma trận 0 cho lỗi từ vựng rỗng. raise chuyển tiếp các lỗi khác.
        if not str(error).startswith("empty vocabulary"):
            raise
        # Ma trận N × N toàn số 0 nghĩa là không có cạnh tương đồng.
        return [[0.0] * len(sentences) for _ in sentences]

    # Tính cosine cho từng cặp câu rồi đổi ma trận thành danh sách Python.
    return cosine_similarity(counts).tolist()
