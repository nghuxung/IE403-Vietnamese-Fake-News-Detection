CLICKBAIT_WORDS = [
    "sốc",
    "khẩn cấp",
    "gây chấn động",
    "không thể tin",
    "bí mật",
    "sự thật",
    "cảnh báo",
    "nguy hiểm",
    "100%",
    "chữa khỏi",
    "thần dược",
    "xem ngay",
    "chia sẻ ngay",
    "trước khi bị xoá",
    "lừa đảo"
]


def extract_features(title, content):

    text = f"{title} {content}"

    text_lower = text.lower()

    exclamation_count = text.count("!")
    question_count = text.count("?")

    uppercase_chars = sum(
        1 for c in text if c.isupper()
    )

    alpha_chars = sum(
        1 for c in text if c.isalpha()
    )

    uppercase_ratio = (
        uppercase_chars / alpha_chars
        if alpha_chars > 0
        else 0
    )

    matched_words = [
        word
        for word in CLICKBAIT_WORDS
        if word in text_lower
    ]

    clickbait_score = (
        len(matched_words)
        / len(CLICKBAIT_WORDS)
    )

    return {
        "text_length": len(text),

        "exclamation_count": exclamation_count,

        "question_count": question_count,

        "uppercase_ratio": round(
            uppercase_ratio,
            4
        ),

        "clickbait_score": round(
            clickbait_score,
            4
        ),

        "matched_clickbait_words": matched_words
    }