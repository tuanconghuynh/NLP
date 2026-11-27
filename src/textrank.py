def calculate_pagerank(similarity_matrix, damping=0.65, tolerance=1e-8):
    """Tính PageRank có trọng số bằng phương pháp lặp lũy thừa."""
    # Số hàng bằng số câu, cũng là số đỉnh của đồ thị.
    count = len(similarity_matrix)
    # Đồ thị rỗng cho từ điển điểm rỗng.
    if not count:
        return {}

    # Mỗi phần tử chứa các cạnh đi ra và xác suất chuyển của một đỉnh.
    transitions = []
    # enumerate cung cấp chỉ số đỉnh nguồn và hàng trọng số của đỉnh đó.
    for source, weights in enumerate(similarity_matrix):
        # Giữ các cạnh có trọng số dương đi đến câu khác. Bỏ cạnh tự nối.
        links = [
            (target, float(weight))
            for target, weight in enumerate(weights)
            if target != source and weight > 0
        ]
        # Cộng trọng số các cạnh đi ra để chuẩn hóa xác suất.
        total = sum(weight for _, weight in links)
        # Chia từng trọng số cho tổng. Với đỉnh cô lập, danh sách rỗng nên không có phép chia.
        transitions.append([(target, weight / total) for target, weight in links])

    # Khởi tạo điểm đồng đều với tổng bằng 1.
    scores = [1 / count] * count
    # Giới hạn 1000 lượt cập nhật. Dấu gạch dưới là biến đếm không được dùng.
    for _ in range(1000):
        # Cộng điểm cũ của các đỉnh không có cạnh đi ra để phân phối lại.
        dangling = sum(score for score, links in zip(scores, transitions) if not links)
        # Mỗi đỉnh nhận phần nhảy ngẫu nhiên và một phần bằng nhau từ các đỉnh cô lập.
        base = (1 - damping + damping * dangling) / count
        # Bắt đầu tính điểm mới từ phần điểm chung của mọi đỉnh.
        updated = [base] * count
        # zip ghép điểm cũ của mỗi đỉnh với danh sách chuyển của chính đỉnh đó.
        for score, links in zip(scores, transitions):
            # Cộng phần điểm truyền qua cạnh. damping là tỉ lệ điểm đi theo các cạnh.
            for target, probability in links:
                updated[target] += damping * score * probability

        # So sánh tổng độ lệch tuyệt đối giữa hai lượt với ngưỡng 10 mũ -8.
        if sum(abs(new - old) for new, old in zip(updated, scores)) < tolerance:
            # Trả điểm dưới dạng từ điển {chỉ số câu: điểm PageRank}.
            return dict(enumerate(updated))
        # Dùng kết quả vừa tính cho lượt lặp kế tiếp.
        scores = updated

    # Báo lỗi nếu chạm giới hạn lặp mà điểm vẫn chưa hội tụ.
    raise RuntimeError("PageRank did not converge")
