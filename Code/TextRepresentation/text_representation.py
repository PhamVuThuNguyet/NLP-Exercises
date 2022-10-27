from collections import defaultdict
import numpy as np
import nltk


def build_vocab(corpus):
    vocab = {}
    counter = 1
    for doc in corpus:
        for sent in doc:
            for word in sent:
                if not word in vocab:
                    vocab.update({word: counter})
                    counter = counter + 1
    return vocab


def one_hot_encoding(vocab):
    embedding_len = len(vocab)
    one_hot_encode = {}
    for k, v in vocab.items():
        vector = np.zeros(embedding_len)
        vector[v - 1] = 1
        one_hot_encode.update({k: vector})

    return one_hot_encode


def bag_of_words(sent, vocab):
    """

    Args:
        sent: Tokenized sentence
        vocab:

    Returns:

    """
    embedding_len = len(vocab)
    count_dict = defaultdict(int)

    bow_vec = np.zeros(embedding_len)

    for word in sent:
        count_dict[word] += 1

    for k, v in count_dict.items():
        bow_vec[vocab[k] - 1] = v

    return bow_vec


def n_grams(sent, n):
    """

    Args:
        sent: Tokenized sentence
        n:

    Returns:

    """
    ngrams = zip(*[sent[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def tf(word, sent):
    N = len(sent)
    occurance = len([w for w in sent if w == word])

    return occurance / N


def df(word, corpus):
    N = len(corpus)
    count = 0
    for doc in corpus:
        for sent in doc:
            for w in sent:
                if w == word:
                    count += 1

    return count


def idf(word, corpus):
    total_documents = len(corpus)

    return np.log(total_documents / (df(word, corpus) + 1))


def tf_idf(sent, corpus, vocab):
    """

    Args:
        sent: Tokenized sent
        corpus:
        vocab:

    Returns:

    """
    tf_idf_vec = np.zeros(len(vocab))
    for word in sent:
        tf_ = tf(word, sent)
        idf_ = idf(word, corpus)

        value = tf_ * idf_
        tf_idf_vec[vocab[word]] = value

    return tf_idf_vec


if __name__ == "__main__":
    corpus = [['Topic sentences are similar to mini thesis statements',
               'Like a thesis statement a topic sentence has a specific main point'],
              ['Whereas the thesis is the main point of the essay',
               'the topic sentence is the main point of the paragraph'],
              ['Like the thesis statement a topic sentence has a unifying function']]

    new_corpus = []
    for doc in corpus:
        tokenized_sent = []
        for sent in doc:
            tokenized_sent.append(nltk.word_tokenize(sent))

        new_corpus.append(tokenized_sent)

    vocab = build_vocab(new_corpus)
    print(len(vocab), vocab)

    one_hot_encoding = one_hot_encoding(vocab)
    print(len(one_hot_encoding), one_hot_encoding)

    print()

    print("BOW of:", new_corpus[0][0])
    print(bag_of_words(new_corpus[0][0], vocab))

    print()

    print("bigram of:", new_corpus[0][0])
    print(n_grams(new_corpus[0][0], 2))

    print()

    print("tf-idf of:", new_corpus[0][0])
    print(tf_idf(new_corpus[0][0], new_corpus, vocab))
