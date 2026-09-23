import string

from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def tokenize_text(text: str, stopwords: list[str]) -> list[str]:
    # Create a translation table that removes punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Normalize the text and remove punctuation
    text = text.lower().translate(translator)

    # Split the text into tokens and remove stop words
    tokens = [
        token for token in text.split()
        if token not in stopwords
    ]

    # Reduce each token to its stem
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def tokenize_term(term: str, stopwords: list[str]) -> str:
    # Tokenize the term using the existing text tokenizer
    tokens = tokenize_text(term, stopwords)

    # A term must produce exactly one token
    if len(tokens) != 1:
        raise ValueError("term must tokenize to exactly one token")

    return tokens[0]