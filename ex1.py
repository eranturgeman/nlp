from collections import defaultdict
import math

import pickle
import spacy
from datasets import load_dataset

BIGRAM_MODEL_SAV = "/mnt/c/Users/dor/Projects/NLP/nlp/bigram_model.sav"

UNIGRAM_MODEL_PATH = "/mnt/c/Users/dor/Projects/NLP/nlp/unigram_model.sav"

nlp = spacy.load("en_core_web_sm")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
Q2_SENTENCE = 'I have a house in'
Q3_SENTENCE_A = 'Brad Pitt was born in Oklahoma'
Q3_SENTENCE_B = 'The actor was born in USA'
LAMBDA_UNIGRAM = 1/3
LAMBDA_BIGRAM = 2/3


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

        return math.log(p) if p != 0 else -math.inf

def default():
    return defaultdict(int)

class BigramModel:
    def __init__(self):
        self.count = 0
        self.freq_single = defaultdict(int)
        self.freq_pairs = defaultdict(default)

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

        return math.log(p) if p != 0 else -math.inf

    def computePerplexity(self, testSet):
        M = 0
        p = 0

        for sentence in testSet.split("\n"):
            p += self.predict(sentence)
            for token in nlp(sentence):
                if(token.is_alpha):
                    M += 1

        return math.exp(-(p/M))

class LerpModel:
    def __init__(self, unigram, bigram):
        self.unigram = unigram
        self.bigram = bigram

    def predict(self, sentence):
        return (LAMBDA_UNIGRAM * self.unigram.predict(sentence)) + (LAMBDA_BIGRAM * self.bigram.predict(sentence))
    def computePerplexity(self, testSet):
        M = 0
        p = 0

        for sentence in testSet.split("\n"):
            p += self.predict(sentence)
            for token in nlp(sentence):
                if(token.is_alpha):
                    M += 1

        return math.exp(-(p/M))

def q2(sentence, lm):
    last_word = str(nlp(sentence)[-1])

    max_word = ""
    max_count = 0
    for word, count in lm.freq_pairs[last_word].items():
        if count > max_count:
            max_count = count
            max_word = word

    return max_word

def q3_a(lm):
    p = lm.predict(Q3_SENTENCE_A)
    print(f"The probability of the sentence: {Q3_SENTENCE_A} is {p}.")
    p = lm.predict(Q3_SENTENCE_B)
    print(f"The probability of the sentence: {Q3_SENTENCE_B} is {p}.")

def q3_b(lm):
    perp = lm.computePerplexity(Q3_SENTENCE_A + "\n" + Q3_SENTENCE_B)
    print(f"The perplexity is: {perp}")

def q4(lm):
    p = lm.predict(Q3_SENTENCE_A)
    print(f"The probability of the sentence: {Q3_SENTENCE_A} is {p}.")
    p = lm.predict(Q3_SENTENCE_B)
    print(f"The probability of the sentence: {Q3_SENTENCE_B} is {p}.")

    perp = lm.computePerplexity(Q3_SENTENCE_A + "\n" + Q3_SENTENCE_B)
    print(f"The perplexity is: {perp}")

def main(train=False):
    if (train):
        print("Q1:")
        unigram = UnigramModel()
        unigram.fit(dataset["text"])
        print("Unigram training completed.")

        with open(UNIGRAM_MODEL_PATH, "wb") as fh:
            pickle.dump(unigram, fh)

        bigram = BigramModel()
        bigram.fit(dataset["text"])
        print("Bigram training completed.")

        with open(BIGRAM_MODEL_SAV, "wb") as fh:
            pickle.dump(bigram, fh)

    else:
        print("Loading models...")
        with open(UNIGRAM_MODEL_PATH, "rb") as fh:
            unigram = pickle.load(fh)

        with open(BIGRAM_MODEL_SAV, "rb") as fh:
            bigram = pickle.load(fh)

    print("Q2:")
    print(f"The most probable word continuation for the sentence: '{Q2_SENTENCE}' is '{q2(Q2_SENTENCE, bigram)}'")

    print("Q3 A:")
    q3_a(bigram)

    print("Q3 B:")
    q3_b(bigram)

    lerp = LerpModel(unigram, bigram)
    print("Q4:")
    q4(lerp)


if __name__ == "__main__":
    main(True)
    # unigram = UnigramModel()
    # unigram.fit("a b c d\ne f a b\na c d f\n")
    # p = unigram.predict("a d")
    # print(p)