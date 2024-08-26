import re


def text_prep(text):
    # Очистка текста от нерелевантных символов
    text = re.sub('[^а-яА-ЯёЁ\s]', '', text)

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Возвращаем обработанный текст
    return text
