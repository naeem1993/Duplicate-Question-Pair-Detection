import numpy as np

def load_embeddings(filepath, embedding_dim=300):
    """
    Generic function to load embeddings from a text file.
    This can handle GloVe, fastText, and Paragram.
    """
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding_vector
    return embeddings_index

def get_embedding_matrix(word_index, embeddings_index, embedding_dim=300):
    """
    Create an embedding matrix where each row corresponds to the word index from the tokenizer.
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
