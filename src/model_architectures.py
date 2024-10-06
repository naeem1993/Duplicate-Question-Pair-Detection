import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Conv1D, MaxPooling1D, Dense, Dropout, \
    GlobalAveragePooling1D, Concatenate


# 1. Bi-LSTM with Attention Model
def build_bilstm_attention(embedding_matrix, input_length=70, lstm_units=256, dropout_rate=0.2, context_vector_size=256,
                           learning_rate=0.001):
    # Input for Question 1
    input_q1 = tf.keras.Input(shape=(input_length,))

    # Input for Question 2
    input_q2 = tf.keras.Input(shape=(input_length,))

    # Embedding Layer with pre-trained embeddings
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
    )

    # Embedding for Question 1 and Question 2
    embedded_q1 = embedding_layer(input_q1)
    embedded_q2 = embedding_layer(input_q2)

    # Bi-LSTM Layer for Question 1 and Question 2
    bilstm_q1 = Bidirectional(LSTM(units=lstm_units, recurrent_dropout=dropout_rate, return_sequences=True))(
        embedded_q1)
    bilstm_q2 = Bidirectional(LSTM(units=lstm_units, recurrent_dropout=dropout_rate, return_sequences=True))(
        embedded_q2)

    # Attention mechanism
    attention_q1 = Dense(context_vector_size, activation='tanh')(bilstm_q1)
    attention_output_q1 = GlobalAveragePooling1D()(attention_q1)

    attention_q2 = Dense(context_vector_size, activation='tanh')(bilstm_q2)
    attention_output_q2 = GlobalAveragePooling1D()(attention_q2)

    # Concatenate the outputs
    concatenated = Concatenate()([attention_output_q1, attention_output_q2])

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(concatenated)

    model = tf.keras.Model(inputs=[input_q1, input_q2], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# 2. Bi-LSTM with Dense Layer Model
def build_bilstm_dense(embedding_matrix, input_length=70, lstm_units=256, dropout_rate=0.2, dense_units=128,
                       learning_rate=0.001):
    # Input for Question 1
    input_q1 = tf.keras.Input(shape=(input_length,))

    # Input for Question 2
    input_q2 = tf.keras.Input(shape=(input_length,))

    # Embedding Layer with pre-trained embeddings
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
    )

    # Embedding for Question 1 and Question 2
    embedded_q1 = embedding_layer(input_q1)
    embedded_q2 = embedding_layer(input_q2)

    # Bi-LSTM Layer for Question 1 and Question 2
    bilstm_q1 = Bidirectional(LSTM(units=lstm_units, dropout=dropout_rate, return_sequences=False))(embedded_q1)
    bilstm_q2 = Bidirectional(LSTM(units=lstm_units, dropout=dropout_rate, return_sequences=False))(embedded_q2)

    # Concatenate Bi-LSTM outputs
    concatenated = Concatenate()([bilstm_q1, bilstm_q2])

    # Dense Layer
    dense_output = Dense(dense_units, activation='relu')(concatenated)
    dropout_output = Dropout(dropout_rate)(dense_output)

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(dropout_output)

    model = tf.keras.Model(inputs=[input_q1, input_q2], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# 3. GRU with Attention Model
def build_gru_attention(embedding_matrix, input_length=70, gru_units=256, dropout_rate=0.2, context_vector_size=256,
                        learning_rate=0.001):
    # Input for Question 1
    input_q1 = tf.keras.Input(shape=(input_length,))

    # Input for Question 2
    input_q2 = tf.keras.Input(shape=(input_length,))

    # Embedding Layer with pre-trained embeddings
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
    )

    # Embedding for Question 1 and Question 2
    embedded_q1 = embedding_layer(input_q1)
    embedded_q2 = embedding_layer(input_q2)

    # GRU Layer for Question 1 and Question 2
    gru_q1 = GRU(units=gru_units, recurrent_dropout=dropout_rate, return_sequences=True)(embedded_q1)
    gru_q2 = GRU(units=gru_units, recurrent_dropout=dropout_rate, return_sequences=True)(embedded_q2)

    # Attention mechanism
    attention_q1 = Dense(context_vector_size, activation='tanh')(gru_q1)
    attention_output_q1 = GlobalAveragePooling1D()(attention_q1)

    attention_q2 = Dense(context_vector_size, activation='tanh')(gru_q2)
    attention_output_q2 = GlobalAveragePooling1D()(attention_q2)

    # Concatenate the outputs
    concatenated = Concatenate()([attention_output_q1, attention_output_q2])

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(concatenated)

    model = tf.keras.Model(inputs=[input_q1, input_q2], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# 4. LSTM with CNN Model
def build_lstm_cnn(embedding_matrix, input_length=70, lstm_units=128, cnn_filters=128, kernel_size=3, pooling_size=2,
                   dropout_rate=0.2, learning_rate=0.001):
    # Input for Question 1
    input_q1 = tf.keras.Input(shape=(input_length,))

    # Input for Question 2
    input_q2 = tf.keras.Input(shape=(input_length,))

    # Embedding Layer with pre-trained embeddings
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
    )

    # Embedding for Question 1 and Question 2
    embedded_q1 = embedding_layer(input_q1)
    embedded_q2 = embedding_layer(input_q2)

    # LSTM Layer for Question 1 and Question 2
    lstm_q1 = LSTM(units=lstm_units, dropout=dropout_rate, return_sequences=True)(embedded_q1)
    lstm_q2 = LSTM(units=lstm_units, dropout=dropout_rate, return_sequences=True)(embedded_q2)

    # CNN Layer for Question 1 and Question 2
    cnn_q1 = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu')(lstm_q1)
    cnn_q1 = MaxPooling1D(pool_size=pooling_size)(cnn_q1)
    cnn_q1 = GlobalAveragePooling1D()(cnn_q1)

    cnn_q2 = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu')(lstm_q2)
    cnn_q2 = MaxPooling1D(pool_size=pooling_size)(cnn_q2)
    cnn_q2 = GlobalAveragePooling1D()(cnn_q2)

    # Concatenate the outputs
    concatenated = Concatenate()([cnn_q1, cnn_q2])

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(concatenated)

    model = tf.keras.Model(inputs=[input_q1, input_q2], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
