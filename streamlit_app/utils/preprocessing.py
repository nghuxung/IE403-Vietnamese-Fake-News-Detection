import re
import unicodedata

def clean_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Xóa HTML
    text = re.sub(r"<.*?>", " ", text)

    # Xóa URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text)

    return text.strip()