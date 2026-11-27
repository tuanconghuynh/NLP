
from src.preprocessing import split_sentences
from src.similarity import calculate_similarity
from src.textrank import calculate_pagerank


def summarize_text(text, ratio=0.3):
    """Tóm tắt văn bản bằng thuật toán TextRank."""

    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= 2:
        return " ".join(sentences)

    # Tính độ tương đồng giữa các câu
    similarity_matrix = calculate_similarity(sentences)

    # Tính điểm PageRank
    scores = calculate_pagerank(similarity_matrix)

    # Xác định số câu cần chọn
    num_sentences = max(
        1,
        min(len(sentences), round(len(sentences) * ratio))
    )

    # Xếp hạng câu theo điểm PageRank
    ranked_sentences = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Chọn các câu có điểm cao nhất
    selected_indices = [
        index
        for index, score in ranked_sentences[:num_sentences]
    ]

    # Sắp xếp theo vị trí xuất hiện ban đầu
    selected_indices.sort()

    # Ghép thành bản tóm tắt
    summary = " ".join(
        sentences[index]
        for index in selected_indices
    )

    return summary
