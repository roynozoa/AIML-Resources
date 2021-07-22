# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

def solution_B4():
    bbc = pd.read_csv('https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/bbc-text.csv')
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    labels = bbc['category'].values
    articles = bbc['text'].values

    train_size = int(len(articles) * training_portion)

    train_articles = articles[0: train_size]
    train_labels = labels[0: train_size]

    validation_articles = articles[train_size:]
    validation_labels = labels[train_size:]

    # train_articles, validation_articles, train_labels, validation_labels = train_test_split(
    #     articles, 
    #     labels, 
    #     train_size=training_portion)

    print(len(train_articles), len(validation_articles), len(train_labels), len(validation_labels), len(labels))

    tokenizer =  Tokenizer(num_words = vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(train_articles)

    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(train_articles) 

    train_padded = pad_sequences(train_seq,maxlen=max_length, truncating=trunc_type) 

    validation_seq = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = pad_sequences(validation_seq,maxlen=max_length, padding=padding_type, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    
    # print(training_label_seq[0], validation_label_seq[0], training_label_seq.shape, validation_label_seq.shape)
    # print(label_tokenizer.word_index)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),

        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        # tf.keras.layers.GRU(128, return_sequences=True),
        # tf.keras.layers.GRU(128),
        #
        # tf.keras.layers.Dense(embedding_dim, activation='relu'),

        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                      patience = 10,
                                                      restore_best_weights=True)
        

    model.fit(train_padded,
              training_label_seq,
              epochs=100,
              validation_data=(validation_padded, validation_label_seq),
              verbose=2)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
