import re
import nltk
from nltk.tokenize import sent_tokenize

# Tải dữ liệu cần thiết
nltk.download("punkt")
nltk.download("punkt_tab")


def clean_text(text):
    """Làm sạch văn bản đầu vào."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text):
    """Tách văn bản thành các câu."""
    text = clean_text(text)

    sentences = sent_tokenize(text, language="english")

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]