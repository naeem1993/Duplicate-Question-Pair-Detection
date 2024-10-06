import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def clean_text(text):
    """
    Clean the text by removing stopwords, punctuation, and converting to lowercase.
    """
    if isinstance(text, str):
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)

        return cleaned_text
    else:
        return ''


def preprocess_text_series(text_series):
    """
    Apply text cleaning to a pandas Series of texts.
    """
    return text_series.apply(clean_text)


def tokenize_texts(texts, max_num_words=50000):
    """
    Tokenize a list of texts and fit a tokenizer.
    """
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return sequences, word_index, tokenizer


def pad_sequences_custom(sequences, maxlen=70):
    """
    Pad sequences to the same length.
    """
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    return padded_sequences


def shuffle_data(X, y):
    """
    Shuffle the data and labels in unison.
    """
    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)
    return np.array(X_shuffled), np.array(y_shuffled)


def preprocess_data(df, text_columns, max_num_words=50000, max_sequence_length=70):
    """
    Full preprocessing pipeline for multiple text columns: clean text, tokenize, pad sequences.
    """
    # Clean text in each specified column
    for column in text_columns:
        df[column] = preprocess_text_series(df[column])

    # Combine texts from all columns for fitting the tokenizer
    combined_texts = df[text_columns[0]].astype(str) + ' ' + df[text_columns[1]].astype(str)

    # Tokenize combined texts
    sequences, word_index, tokenizer = tokenize_texts(combined_texts.tolist(), max_num_words)

    # Separate sequences for each text column
    sequences_q1 = tokenizer.texts_to_sequences(df[text_columns[0]].tolist())
    sequences_q2 = tokenizer.texts_to_sequences(df[text_columns[1]].tolist())

    # Pad sequences
    padded_sequences_q1 = pad_sequences_custom(sequences_q1, maxlen=max_sequence_length)
    padded_sequences_q2 = pad_sequences_custom(sequences_q2, maxlen=max_sequence_length)

    return padded_sequences_q1, padded_sequences_q2, word_index, tokenizer
