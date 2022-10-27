import os

import underthesea as uts

from keras.utils import np_utils, pad_sequences
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

from sklearn.metrics.pairwise import euclidean_distances

import numpy as np


# 1. Build corpus vocab
def build_vocab(corpus):
    vocab = {}
    i = 1
    for doc in corpus:
        for sent in doc:
            for word in sent:
                if not word in vocab:
                    vocab[word] = i
                    i += 1

    return vocab


# 2. Build CBOW generator (context, target)
def generate_context_target_pair(corpus, window_size, vocab):
    context_length = window_size * 2
    for doc in corpus:
        for sent in doc:
            sentence_length = len(sent)
            for index, word in enumerate(sent):
                context_words = []
                label_word = []
                start = index - window_size
                end = index + window_size + 1

                context_words.append([vocab[sent[i]]
                                      for i in range(start, end)
                                      if 0 <= i < sentence_length
                                      and i != index])
                label_word.append(vocab[word])
                x = pad_sequences(context_words, maxlen=context_length)
                y = np_utils.to_categorical(label_word, len(vocab))
                yield (x, y)


# 3. Build CBOW model architecture
def model(vocab_size, embed_size, window_size):
    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size * 2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='adam')

    # view model summary
    print(cbow.summary())

    return cbow


# 4. Train model
def train(model, corpus, epochs, window_size, vocab):
    for epoch in range(1, epochs):
        loss = 0.
        i = 0
        for x, y in generate_context_target_pair(corpus, window_size, vocab):
            i += 1
            loss += model.train_on_batch(x, y)
            if i % 100000 == 0:
                print('Processed {} (context, word) pairs'.format(i))

        print('Epoch:', epoch, '\tLoss:', loss)
        print()


# 5. Get Word Embedding


if __name__ == "__main__":
    folder = "../../underthesea_data/corpus/"

    corpus = []

    for f in os.listdir(folder):
        with open(os.path.join(folder, f), encoding="utf-8") as file:
            corpus.append(uts.sent_tokenize(file.read()))

    final_corpus = []
    for e in corpus:
        temp = []
        for s in e:
            temp.append(uts.word_tokenize(s))
        final_corpus.append(temp)

    vocab_dict = build_vocab(final_corpus)

    model = model(len(vocab_dict), 100, 2)
    train(model, final_corpus, 1, 2, vocab_dict)

    embedding_vects = model.get_weights()[0]
    print(embedding_vects[vocab_dict["dân"]])

    # compute pairwise distance matrix
    distance_matrix = euclidean_distances(embedding_vects)
    print(distance_matrix.shape)

    id2word = {v: k for k, v in vocab_dict.items()}

    # view contextually similar words
    similar_words = {
        search_term: [id2word[idx] for idx in distance_matrix[vocab_dict[search_term] - 1].argsort()[1:6] + 1]
        for search_term in ['dân', 'xã', 'hội']}

    print(similar_words)
