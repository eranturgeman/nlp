from collections import defaultdict
import math

import spacy
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

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
        self.freq_pairs = defaultdict(int)

    def fit(self, texts):
        for text in texts:
            for line in text.split('\n'):
                doc = nlp(line)
                prev = "START"
                for token in doc:
                    if (token.is_alpha):
                        self.count += 1
                        self.freq_single[prev] += 1
                        self.freq_pairs[(prev, token.lemma_)] += 1
                        prev = token.lemma_

    def predict(self, sentence):
        p = 1
        prev = "START"
        for token in nlp(sentence):
            if (token.is_alpha):
                m = self.freq_single[prev]
                if m == 0:
                    return 0

                p *= (self.freq_pairs[(prev, token.lemma_)] / m)
                prev = token.lemma_

        return p
        # return math.log(p, 10) if p != 0 else 0

def main():
    # unigram = UnigramModel()
    # unigram.fit(["Train maximum-likelihood unigram and bigram language\ncompute the probability of the following two sentences\n"])
    # unigram.fit(dataset["text"])
    # print(unigram.predict("language language"))

    bigram = BigramModel()
    bigram.fit(["Train the bigram language\ncompute the two sentences\n"])
    print(bigram.predict("Train the"))


if __name__ == "__main__":
    main()