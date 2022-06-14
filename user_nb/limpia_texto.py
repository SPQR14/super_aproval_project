import unicodedata
import re


def clean_text_eng(text, pattern="[^a-zA-Z0-9 ]"):
    cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    cleaned_text = re.sub(pattern, " ", cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.lower().strip().split())
    return cleaned_text


def clean_text_spa(text, pattern="[^a-zA-Z-0-9ñÑáéíóúüÁÉÍÓÚÜ\s]+"):
    cleaned_text = unicodedata.normalize('NFD', text).encode('utf-8', 'ignore')
    cleaned_text = re.sub(pattern, " ", cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.lower().strip().split())
    return cleaned_text


