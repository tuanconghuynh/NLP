# re cung cấp biểu thức chính quy để chuẩn hóa khoảng trắng.
import re
# sent_tokenize dùng mô hình tách câu Punkt của NLTK.
from nltk.tokenize import sent_tokenize


def clean_text(text):
    """Làm sạch văn bản đầu vào."""
    # Mẫu \\s+ khớp một hoặc nhiều ký tự trắng liên tiếp, gồm dấu xuống dòng và tab.
    text = re.sub(r'\s+', ' ', text)
    # Bỏ khoảng trắng thừa ở đầu và cuối văn bản.
    return text.strip()


def split_sentences(text):
    """Tách văn bản thành các câu."""
    # Chuẩn hóa khoảng trắng trước khi tách câu.
    text = clean_text(text)
    # Chuỗi rỗng sau khi làm sạch tương ứng với danh sách không có câu.
    if not text:
        # Giữ các câu có nội dung và bỏ khoảng trắng hai đầu từng câu.
        return []

    # Tách câu tiếng Anh bằng tài nguyên punkt_tab đã cài.
    sentences = sent_tokenize(text, language="english")

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
