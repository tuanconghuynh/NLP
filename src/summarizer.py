# Nhập ba bước xử lý là tách câu, tính độ tương đồng và xếp hạng PageRank.
from src.preprocessing import split_sentences
from src.similarity import calculate_similarity
from src.textrank import calculate_pagerank


def sentence_budget(sentence_count, ratio):
    """Tính số câu cần giữ theo tỉ lệ trong khoảng (0, 1]."""
    # Chỉ chấp nhận tỉ lệ lớn hơn 0 và tối đa bằng 1.
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in the interval (0, 1]")
    # Văn bản có câu giữ tối thiểu 1 câu. Văn bản rỗng giữ 0 câu.
    return max(1, round(sentence_count * ratio)) if sentence_count else 0


def summarize_sentences(sentences, num_sentences):
    """Chọn các câu bằng TextRank và giữ thứ tự xuất hiện."""
    # Khi giữ toàn bộ câu, ghép trực tiếp vì việc xếp hạng không đổi kết quả.
    if num_sentences == len(sentences):
        return " ".join(sentences)

    # Ma trận vừa là trọng số đồ thị, vừa dùng để đo độ trùng khi chọn câu.
    similarity_matrix = calculate_similarity(sentences)

    # Tính điểm PageRank
    scores = calculate_pagerank(similarity_matrix)

    # MMR cân bằng điểm PageRank và độ trùng lặp với các câu đã chọn.
    # Chia điểm PageRank cho giá trị lớn nhất để điểm quan trọng nằm trong khoảng [0, 1].
    relevance_scale = max(scores.values())
    # Các ứng viên ban đầu là chỉ số từ 0 đến N - 1.
    remaining = list(range(len(sentences)))
    # Ban đầu chưa chọn câu nào nên mức trùng lặp của mỗi ứng viên bằng 0.
    redundancy = [0.0] * len(sentences)
    # Lưu chỉ số các câu được chọn, riêng với nội dung câu gốc.
    selected_indices = []
    # Mỗi lượt chọn một câu cho đến khi đủ ngân sách.
    for _ in range(num_sentences):
        # key xác định tiêu chí so sánh. Điểm MMR bằng PageRank chuẩn hóa trừ hai lần mức trùng.
        index = max(
            remaining,
            key=lambda candidate: scores[candidate] / relevance_scale
            - 2 * redundancy[candidate],
        )
        # Lưu ứng viên thắng lượt này rồi rút nó khỏi danh sách chờ.
        selected_indices.append(index)
        remaining.remove(index)
        # Cập nhật mức trùng lớn nhất giữa ứng viên và bất kỳ câu nào đã chọn.
        for candidate in remaining:
            redundancy[candidate] = max(
                redundancy[candidate], similarity_matrix[index][candidate]
            )

    # Sắp xếp theo vị trí xuất hiện ban đầu
    selected_indices.sort()

    # Ghép thành bản tóm tắt
    summary = " ".join(
        sentences[index]
        for index in selected_indices
    )

    return summary


def summarize_text(text, ratio=0.3):
    """Tóm tắt văn bản bằng thuật toán TextRank."""
    # Nhận một chuỗi đầu vào và tách thành danh sách các câu.
    sentences = split_sentences(text)
    # Tính ngân sách từ số câu rồi gọi hàm chọn câu.
    return summarize_sentences(sentences, sentence_budget(len(sentences), ratio))
