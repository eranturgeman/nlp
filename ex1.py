from collections import defaultdict
import math

import pickle
import spacy
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
Q2_SENTENCE = 'I have a house in'

class UnigramModel:
    def __init__(self):
        self.count = 0
        self.freq = defaultdict(int)

    def fit(self, texts):
        for text in texts:
            doc = nlp(text)
            for token in doc:
                if (token.is_alpha):
                    self.count += 1
                    self.freq[token.lemma_] += 1

    def predict(self, sentence):
        p = 1
        for token in nlp(sentence):
            if (token.is_alpha):
                p *= (self.freq[token.lemma_] / self.count)

        return p
        # return math.log(p, 10) if p != 0 else 0

class BigramModel:
    def __init__(self):
        self.count = 0
        self.freq_single = defaultdict(int)
        self.freq_pairs = defaultdict(lambda: defaultdict(int))

    def fit(self, texts):
        for text in texts:
            for line in text.split('\n'):
                doc = nlp(line)
                prev = "START"
                for token in doc:
                    if (token.is_alpha):
                        self.count += 1
                        self.freq_single[prev] += 1
                        self.freq_pairs[prev][token.lemma_] += 1
                        prev = token.lemma_

    def predict(self, sentence):
        p = 1
        prev = "START"
        for token in nlp(sentence):
            if (token.is_alpha):
                m = self.freq_single[prev]
                if m == 0:
                    return 0

                p *= (self.freq_pairs[prev][token.lemma_] / m)
                prev = token.lemma_

        return p
        # return math.log(p, 10) if p != 0 else 0


def q2(sentence, lm):
    last_word = nlp(sentence)[-1]

    max_word = ""
    max_count = 0
    for word, count in lm.freq_pairs[last_word].items():
        if count > max_count:
            max_count = count
            max_word = word

    return max_word

def main(train=False):
    # unigram = UnigramModel()
    # unigram.fit(["Train maximum-likelihood unigram and bigram language\ncompute the probability of the following two sentences\n"])
    # unigram.fit(dataset["text"])
    # print(unigram.predict("language language"))
    #
    # bigram = BigramModel()
    # bigram.fit(["a b c d\na a b c\nc b d e\ne f f g"])
    # print(bigram.predict("a b c")) #
    if (train):
        print("Q1:")
        unigram = UnigramModel()
        unigram.fit(dataset["text"])
        print("Unigram training completed.")

        with open("/mnt/c/Users/dor/Projects/NLP/nlp/unigram_model.sav", "wb") as fh:
            pickle.dump(unigram, fh)

        bigram = BigramModel()
        bigram.fit(dataset["text"])
        print("Bigram training completed.")

        with open("/mnt/c/Users/dor/Projects/NLP/nlp/bigram_model.sav", "wb") as fh:
            pickle.dump(bigram, fh)

    else:
        print("Loading models...")
        with open("/mnt/c/Users/dor/Projects/NLP/nlp/unigram_model.sav", "wb") as fh:
            unigram = pickle.load(fh)

        with open("/mnt/c/Users/dor/Projects/NLP/nlp/bigram_model.sav", "wb") as fh:
            bigram = pickle.load(fh)

    print("Q2:")
    print(q2(Q2_SENTENCE, bigram))

    print("Q3:")


if __name__ == "__main__":
    main(True)