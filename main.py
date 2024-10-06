import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data_preprocessing import preprocess_data
from src.feature_extraction import load_embeddings, get_embedding_matrix
from src.model_training import (
    train_gru_attention,
    train_bilstm_attention,
    train_bilstm_dense,
    train_lstm_cnn
)
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

# Import resampling techniques
from src.resampling.adasyn import apply_adasyn
from src.resampling.ros import apply_ros
from src.resampling.smote import apply_smote
from src.resampling.smote_tomek import apply_smote_tomek
from src.resampling.original_imbalanced import use_original_imbalanced


def check_gpu():
    # Check if TensorFlow can see the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs found: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU Name: {gpu}")
    else:
        print("No GPU found. Running on CPU.")


def main():
    check_gpu()  # Check if GPU is detected

    # Load raw data (adjust path as needed)
    df = pd.read_csv('data/raw/train.csv')
    df = df[:5000]

    # Preprocess data
    text_columns = ['question1', 'question2']
    X_q1, X_q2, word_index, tokenizer = preprocess_data(df, text_columns)

    # Split data into train and test sets
    X_train_q1, X_test_q1, y_train, y_test = train_test_split(X_q1, df['is_duplicate'], test_size=0.2)
    X_train_q2, X_test_q2 = train_test_split(X_q2, test_size=0.2)

    # Resampling technique selection
    resampling_technique = 'adasyn'  # Options: 'adasyn', 'ros', 'smote', 'smote_tomek', 'original'

    if resampling_technique == 'adasyn':
        print("Applying ADASYN resampling")
        X_train_q1, X_train_q2, y_train = apply_adasyn(X_train_q1, X_train_q2, y_train)
    elif resampling_technique == 'ros':
        print("Applying Random Over Sampling (ROS)")
        X_train_q1, X_train_q2, y_train = apply_ros(X_train_q1, X_train_q2, y_train)
    elif resampling_technique == 'smote':
        print("Applying SMOTE resampling")
        X_train_q1, X_train_q2, y_train = apply_smote(X_train_q1, X_train_q2, y_train)
    elif resampling_technique == 'smote_tomek':
        print("Applying SMOTE-Tomek resampling")
        X_train_q1, X_train_q2, y_train = apply_smote_tomek(X_train_q1, X_train_q2, y_train)
    elif resampling_technique == 'original':
        print("Using original imbalanced data without resampling")
        X_train_q1, X_train_q2, y_train = use_original_imbalanced(X_train_q1, X_train_q2, y_train)
    else:
        raise ValueError("Invalid resampling technique specified.")

    # Choose which embeddings to use
    embedding_type = 'glove'  # Options: 'glove', 'fasttext', 'paragram'

    if embedding_type == 'glove':
        embeddings_path = 'data/embeddings/glove.6B.300d.txt'
        print("Using GloVe embeddings")
    elif embedding_type == 'fasttext':
        embeddings_path = 'data/embeddings/wiki-news-300d-1M.vec'
        print("Using fastText embeddings")
    elif embedding_type == 'paragram':
        embeddings_path = 'data/embeddings/paragram_300_sl999.txt'
        print("Using Paragram embeddings")
    else:
        raise ValueError("Invalid embedding type specified.")

    # Load the chosen embeddings
    embeddings = load_embeddings(embeddings_path)

    # Create the embedding matrix
    embedding_matrix = get_embedding_matrix(word_index, embeddings)

    # Model selection
    model_type = 'gru_attention'  # Options: 'gru_attention', 'bilstm_attention', 'bilstm_dense', 'lstm_cnn'

    if model_type == 'gru_attention':
        print("Training GRU with Attention model")
        model, history = train_gru_attention(
            embedding_matrix, [X_train_q1, X_train_q2], [X_test_q1, X_test_q2], y_train, y_test)
    elif model_type == 'bilstm_attention':
        print("Training Bi-LSTM with Attention model")
        model, history = train_bilstm_attention(
            embedding_matrix, [X_train_q1, X_train_q2], [X_test_q1, X_test_q2], y_train, y_test)
    elif model_type == 'bilstm_dense':
        print("Training Bi-LSTM with Dense Layer model")
        model, history = train_bilstm_dense(
            embedding_matrix, [X_train_q1, X_train_q2], [X_test_q1, X_test_q2], y_train, y_test)
    elif model_type == 'lstm_cnn':
        print("Training LSTM with CNN model")
        model, history = train_lstm_cnn(
            embedding_matrix, [X_train_q1, X_train_q2], [X_test_q1, X_test_q2], y_train, y_test)
    else:
        raise ValueError("Invalid model type specified.")

    # Evaluate model performance
    accuracy, precision, recall, f1 = evaluate_model(model, [X_test_q1, X_test_q2], y_test)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Plot confusion matrix and ROC curve
    y_pred_prob = model.predict([X_test_q1, X_test_q2])
    plot_confusion_matrix(y_test, (y_pred_prob > 0.5).astype("int32"))
    plot_roc_curve(y_test, y_pred_prob)


if __name__ == "__main__":
    main()
