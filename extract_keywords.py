from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


def extract_keywords(texts):
    # Объединение всех текстов в одну большую строку
    all_texts = ' '.join(texts)

    # Токенизация текста
    tokens = word_tokenize(all_texts)

    # Приведение слов к нижнему регистру
    tokens = [word.lower() for word in tokens]

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Подсчет частоты слов
    word_freq = Counter(tokens)

    # Возвращение словаря с ключевыми словами и их частотами
    return dict(word_freq)
