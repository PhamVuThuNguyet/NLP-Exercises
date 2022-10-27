import underthesea as uts
import gensim
import numpy as np
import unicodedata
import itertools
from langdetect import detect_langs
import nltk

path = "../DataScraping/data/iPhone 14 lộ toàn bộ giá bán.txt"

with open(path, "r+", encoding="utf-8") as file:
    text = file.read()

# Unicode normalization
print("Unicode Normalization of ●: ", hex(ord(unicodedata.normalize("NFKD", "●"))))
print("")

# Sentence Tokenization
sentTokenize = uts.sent_tokenize(text)
with open("./sent_tokenize.txt", "a+", encoding="utf-8") as file:
    for sent in sentTokenize:
        file.write(sent)

print("Sentence Tokenization: ", sentTokenize)
print("")

# Remove Stopwords
with open("./stopwords.txt", "r+", encoding="utf-8") as file:
    stopwords = file.read().split("\n")

print("Remove Stopword")
print("Before: " + sentTokenize[0])
sentSplit = uts.word_tokenize(sentTokenize[0])
sentNoStopWords = [w for w in sentSplit if not w in stopwords]
print("After:", " ".join(sentNoStopWords))

print("")

# Word Tokenization
wordTokenize = []
for sent in sentTokenize:
    wordTokenize.append(uts.word_tokenize(sent))

print("Word Tokenize")
print(wordTokenize)
print("")

# Count Frequency and Sort
dictionary = gensim.corpora.Dictionary(wordTokenize)

flatten_token_list = list(itertools.chain.from_iterable(wordTokenize))
flatten_corpus = [dictionary.doc2bow(flatten_token_list)]

freq_dict = {}
for i in range(len(flatten_corpus[0])):
    id, freq = flatten_corpus[0][i]
    freq_dict[dictionary[id]] = freq

print("Frequency of tokens")
print(dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)))
print("")

# Stemming & Lemma
print("Stemming")
words = ["program", "programs", "programmer", "programming", "programmers"]
for w in words:
    print(w, ": ", nltk.stem.PorterStemmer().stem(w))

print("Lemma")
print("rocks :", nltk.stem.WordNetLemmatizer().lemmatize("rocks"))
print("corpora :", nltk.stem.WordNetLemmatizer().lemmatize("corpora"))
print("better :", nltk.stem.WordNetLemmatizer().lemmatize("better", pos="a"))

print("")

# Lang Detect
print("Lang Detect")
print(detect_langs(text=sentTokenize[0]))