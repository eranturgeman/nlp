import itertools

import numpy as np
import nltk
from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

TRAIN_SET_SIZE = 0.9
EPOCH_NUM = 2

class MSTParser:
    def __init__(self, sentences):
        self.split_train_test(sentences)
        self.get_words_and_tags()

    def split_train_test(self, sentences):
        n = len(sentences)
        train_idx = round(n * TRAIN_SET_SIZE)

        self.train_set = sentences[:train_idx]
        self.test_set = sentences[train_idx:]

    def get_words_and_tags(self):
        words = set()
        pos_tags = set()
        for sent in self.train_set:
            for node in sent.nodes:
                words.add(sent.nodes[node]['word'])
                pos_tags.add(sent.nodes[node]['tag'])

        self.words = sorted(list(words))
        self.pos_tags = sorted(list(pos_tags))

        self.words_mapping = {(a, b): i for (i, (a, b)) in enumerate(itertools.product(words, repeat=2))}
        words_offset = len(self.words_mapping)
        self.pos_mapping = {(a, b): (i + words_offset) for (i, (a, b)) in enumerate(itertools.product(self.pos_tags, repeat=2))}

        self.feature_vec_size = words_offset + len(self.pos_mapping)

    def create_feature_vector(self, first_word, second_word, first_pos, second_pos):
        result = np.zeros(self.feature_vec_size)

        result[self.words_mapping[(first_word, second_word)]] += 1
        result[self.pos_mapping[(first_pos, second_pos)]] += 1

        return result

    def get_arcs(self, sent, w):
        arcs = []
        for i in range(len(sent.nodes) - 1):
            word1 = sent.nodes[i]['word']
            word2 = sent.nodes[i + 1]['word']
            word1_pos = sent.nodes[i]['tag']
            word2_pos = sent.nodes[i + 1]['tag']
            pair_feat_vector = self.create_feature_vector(word1, word2, word1_pos, word2_pos)
            pair_score = np.dot(pair_feat_vector, w)
            arcs.append((word1, word2, pair_score))
        return arcs

    def train(self):
        w_sum = np.zeros(self.feature_vec_size)
        w = np.zeros(self.feature_vec_size)

        #iterate NUM_EPOCHS
        #for each epoch iterate over all sentences
        #for each sentence iterate over all pairs of following words
        #calc score for each pair and create an arc for each pair: (word1, word2, score)
        #keep all arcs of a sentence and send them to Edmonds to get T' for this sentence
        #update wights add to w_sum
        #normalize w_sum


        for r in range(EPOCH_NUM):
            for sent in self.train_set:
                mst = min_spanning_arborescence_nx(self.get_arcs(sent, w), None)






def main():
    nltk.download('dependency_treebank')
    sentences = dependency_treebank.parsed_sents()
    mst_parser = MSTParser(sentences)



if __name__ == "__main__":
    main()
