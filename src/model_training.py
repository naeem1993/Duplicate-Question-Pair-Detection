import tensorflow as tf
from src.model_architectures import (
    build_gru_attention,
    build_bilstm_attention,
    build_bilstm_dense,
    build_lstm_cnn
)


def train_gru_attention(embedding_matrix, X_train, X_test, y_train, y_test, epochs=160, batch_size=512):
    """
    Trains a GRU with Attention model and returns the trained model and training history.
    """
    with tf.device('/GPU:0'):  # Explicitly place the operation on the first GPU
        model = build_gru_attention(embedding_matrix=embedding_matrix)

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )

    return model, history


def train_bilstm_attention(embedding_matrix, X_train, X_test, y_train, y_test, epochs=160, batch_size=512):
    """
    Trains a Bi-LSTM with Attention model and returns the trained model and training history.
    """
    with tf.device('/GPU:0'):
        model = build_bilstm_attention(embedding_matrix=embedding_matrix)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
    return model, history


def train_bilstm_dense(embedding_matrix, X_train, X_test, y_train, y_test, epochs=160, batch_size=512):
    """
    Trains a Bi-LSTM with Dense Layer model and returns the trained model and training history.
    """
    with tf.device('/GPU:0'):
        model = build_bilstm_dense(embedding_matrix=embedding_matrix)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
    return model, history


def train_lstm_cnn(embedding_matrix, X_train, X_test, y_train, y_test, epochs=160, batch_size=512):
    """
    Trains an LSTM with CNN model and returns the trained model and training history.
    """
    with tf.device('/GPU:0'):
        model = build_lstm_cnn(embedding_matrix=embedding_matrix)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
    return model, history
