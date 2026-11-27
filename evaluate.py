# Các thư viện chuẩn xử lý tham số dòng lệnh, JSON, token, số đếm và trung bình.
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

# PorterStemmer rút gọn các dạng từ tiếng Anh khi chấm ROUGE.
from nltk.stem import PorterStemmer

from src.preprocessing import split_sentences
from src.summarizer import sentence_budget, summarize_sentences


# Đặt thứ tự các chỉ số để tính và in bảng nhất quán.
METRICS = ("rouge1", "rouge2", "rougeL")
STEMMER = PorterStemmer()


def tokenize(text):
    # Chuyển về chữ thường, lấy các cụm chữ hoặc số và stemming token dài hơn 3 ký tự.
    return [STEMMER.stem(token) if len(token) > 3 else token
            for token in re.findall(r"[a-z0-9]+", text.lower())]


def f1(overlap, reference_count, prediction_count):
    # Rút gọn trung bình điều hòa của precision và recall. Khi không có phần trùng, trả 0.
    return 2 * overlap / (reference_count + prediction_count) if overlap else 0.0


def rouge_scores(reference, prediction):
    # Token hóa đáp án và bản sinh theo cùng một quy tắc.
    reference_tokens = tokenize(reference)
    prediction_tokens = tokenize(prediction)
    scores = {}
    # ROUGE-1 đếm từng token, ROUGE-2 đếm cặp token liền nhau.
    for size, metric in enumerate(METRICS[:2], start=1):
        # Cắt dãy token theo độ lệch rồi dùng zip để ghép thành n-gram. Counter đếm số lần xuất hiện.
        ref_ngrams = Counter(zip(*(reference_tokens[offset:] for offset in range(size))))
        pred_ngrams = Counter(zip(*(prediction_tokens[offset:] for offset in range(size))))
        # Phép & giữa Counter lấy số đếm nhỏ hơn của mỗi n-gram chung.
        scores[metric] = f1(
            sum((ref_ngrams & pred_ngrams).values()),
            ref_ngrams.total(),
            pred_ngrams.total(),
        )

    # Hàng đầu của bảng quy hoạch động cho dãy con chung dài nhất (LCS).
    previous = [0] * (len(prediction_tokens) + 1)
    # Mỗi vòng ngoài xét thêm một token của bản tham chiếu.
    for ref_token in reference_tokens:
        # Cột đầu bằng 0 vì bản sinh có tiền tố rỗng.
        current = [0]
        # index bắt đầu từ 1 vì cột 0 dành cho tiền tố rỗng.
        for index, pred_token in enumerate(prediction_tokens, start=1):
            # Hai token giống nhau lấy ô chéo cộng 1, còn lại lấy giá trị lớn hơn từ ô trên và ô trái.
            current.append(previous[index - 1] + 1 if ref_token == pred_token
                           else max(previous[index], current[-1]))
        # Sau mỗi hàng, chỉ giữ hàng vừa tính để tiết kiệm bộ nhớ.
        previous = current
    # Ô cuối là độ dài LCS. Dùng độ dài này làm phần trùng trong F1 của ROUGE-L.
    scores["rougeL"] = f1(previous[-1], len(reference_tokens), len(prediction_tokens))
    return scores


def evaluate(dataset_path, ratio):
    # Kiểm tra ratio ngay cả khi tập dữ liệu rỗng.
    sentence_budget(0, ratio)
    # Mỗi phương pháp có danh sách điểm riêng cho từng tài liệu.
    scores = {"TextRank": [], "Lead-K": []}

    # Đọc JSONL từng dòng. utf-8-sig cũng xử lý được dấu BOM ở đầu tệp.
    with dataset_path.open(encoding="utf-8-sig") as dataset:
        # Đếm dòng từ 1 để thông báo lỗi chỉ đúng vị trí.
        for line_number, line in enumerate(dataset, start=1):
            # Bỏ qua dòng trắng.
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: {error.msg}") from error
            # Yêu cầu bản ghi dạng dict và hai trường text, reference đều là chuỗi có nội dung.
            if not isinstance(record, dict) or any(
                not isinstance(record.get(field), str) or not record[field].strip()
                for field in ("text", "reference")
            ):
                raise ValueError(f"Line {line_number}: expected nonempty text and reference")

            # Hai phương pháp dùng cùng danh sách câu và cùng ngân sách K.
            sentences = split_sentences(record["text"])
            k = sentence_budget(len(sentences), ratio)
            # TextRank dùng đồ thị. Lead-K lấy các phần tử từ đầu danh sách đến trước chỉ số k.
            predictions = {
                "TextRank": summarize_sentences(sentences, k),
                "Lead-K": " ".join(sentences[:k]),
            }
            # Chấm riêng từng bản sinh với cùng bản tham chiếu của tài liệu.
            for method, prediction in predictions.items():
                scores[method].append(rouge_scores(record["reference"], prediction))

    # Hai phương pháp chấm cùng số tài liệu nên lấy số phần tử từ một danh sách.
    count = len(scores["TextRank"])
    if not count:
        raise ValueError("Dataset contains no documents")
    # Trả số tài liệu và điểm F1 trung bình riêng theo từng phương pháp, từng chỉ số.
    return count, {
        method: {
            metric: mean(document[metric] for document in documents)
            for metric in METRICS
        }
        for method, documents in scores.items()
    }


def main():
    # Khai báo đường dẫn JSONL bắt buộc và tỉ lệ câu tùy chọn có mặc định 0,3.
    parser = argparse.ArgumentParser(description="Compare TextRank and Lead-K using ROUGE F1.")
    parser.add_argument("dataset", type=Path, help="JSONL file with text and reference fields")
    parser.add_argument("--ratio", type=float, default=0.3)
    # Phân tích tham số từ terminal và chuyển chúng sang kiểu đã khai báo.
    args = parser.parse_args()
    try:
        # Chạy đánh giá, rồi dùng parser.error để hiện lỗi đầu vào nếu có.
        count, scores = evaluate(args.dataset, args.ratio)
    except (OSError, ValueError, LookupError) as error:
        parser.error(str(error))

    # In số tài liệu, tỉ lệ câu và tiêu đề bảng.
    print(f"Documents: {count} | Sentence ratio: {args.ratio}")
    print(f"{'Method':<12} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9}")
    for method, metrics in scores.items():
        # Định dạng mỗi điểm rộng 9 ký tự và có 4 chữ số sau dấu thập phân.
        values = " ".join(f"{metrics[metric]:9.4f}" for metric in METRICS)
        print(f"{method:<12} {values}")


# Chạy giao diện dòng lệnh khi gọi trực tiếp evaluate.py.
if __name__ == "__main__":
    main()
