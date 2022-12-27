import itertools
import math
import random
import collections
import numpy as np
import nltk
from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

Arc = collections.namedtuple('Arc', ['head', 'tail', 'weight', 'head_pos', 'tail_pos'])
TRAIN_SET_SIZE = 0.9
EPOCH_NUM = 2
ROOT_WORD_LEN = 15
LEARNING_RATE = 1

class MSTParser:
    def __init__(self, sentences):
        sentences = sentences[:500] #todo delete
        self.split_train_test(sentences)
        self.create_random_root_word(ROOT_WORD_LEN)
        self.get_words_and_tags()
        self.weights = self.train()

    def split_train_test(self, sentences):
        n = len(sentences)
        train_idx = math.ceil(n * TRAIN_SET_SIZE)

        self.train_set = sentences[:train_idx]
        self.test_set = sentences[train_idx:]

    def create_random_root_word(self, length):
        res = ""
        for _ in range(length):
            res += chr(random.randint(33, 47))

        self.root_word = res

    def get_words_and_tags(self):
        words = set()
        pos_tags = set()
        for sent in self.train_set:
            for node in sent.nodes:
                if not sent.nodes[node]['word']:
                    sent.nodes[node]['word'] = self.root_word
                words.add(sent.nodes[node]['word'])
                pos_tags.add(sent.nodes[node]['tag'])

        self.words = sorted(list(words))
        self.pos_tags = sorted(list(pos_tags))

        self.words_mapping = {(a, b): i for (i, (a, b)) in enumerate(itertools.product(self.words, repeat=2))}
        words_offset = len(self.words_mapping)
        self.pos_mapping = {(a, b): (i + words_offset) for (i, (a, b)) in enumerate(itertools.product(self.pos_tags, repeat=2))}

        self.feature_vec_size = words_offset + len(self.pos_mapping)

    def create_feature_vector(self, first_word, second_word, first_pos, second_pos):
        result = np.zeros(self.feature_vec_size)

        if (first_word, second_word) in self.words_mapping:
            result[self.words_mapping[(first_word, second_word)]] += 1
        if (first_pos, second_pos) in self.pos_mapping:
            result[self.pos_mapping[(first_pos, second_pos)]] += 1

        return result

    def create_feature_vector_from_mst(self, mst):
        result = np.zeros(self.feature_vec_size)

        for arc in mst.values():
            result[self.words_mapping[(arc.head, arc.tail)]] += 1
            result[self.pos_mapping[(arc.head_pos, arc.tail_pos)]] += 1

        return result

    def create_feature_for_gold_standard(self, sent):
        result = np.zeros(self.feature_vec_size)

        for node in sent.nodes:
            word = sent.nodes[node]['word']
            pos = sent.nodes[node]['tag']
            parent_idx = sent.nodes[node]['head']
            if not parent_idx:
                continue
            parent_word = sent.nodes[parent_idx]['word']
            parent_pos = sent.nodes[parent_idx]['tag']
            result[self.words_mapping[(parent_word, word)]] += 1
            result[self.pos_mapping[(parent_pos, pos)]] += 1

        return result

    def get_arcs(self, sent, w):
        arcs = []
        for i in range(len(sent.nodes)):
            word1 = sent.nodes[i]['word']
            word1_pos = sent.nodes[i]['tag']
            if word1_pos == "TOP":
                sent.nodes[i]['word'] = self.root_word
                word1 = self.root_word

            for j in range(len(sent.nodes)):
                word2 = sent.nodes[j]['word']
                word2_pos = sent.nodes[j]['tag']
                if j == i or word2_pos == "TOP":
                    continue

                pair_feat_vector = self.create_feature_vector(word1, word2, word1_pos, word2_pos)
                pair_score = np.dot(pair_feat_vector, w)
                arcs.append(Arc(word1, word2, -pair_score, word1_pos, word2_pos))  # todo make sure the invert sign achieves what we want
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
            k = 0  # todo del
            for sent in self.train_set:
                print(f"epoch {r} sentence {k}") #todo del
                mst = min_spanning_arborescence_nx(self.get_arcs(sent, w), None)
                mst_feature_vector = self.create_feature_vector_from_mst(mst)
                gold_standard_feature_vector = self.create_feature_for_gold_standard(sent)

                w = w + (gold_standard_feature_vector - mst_feature_vector) #learning rate = 1
                w_sum += w
                k += 1 #todo del

        return w_sum / (EPOCH_NUM * len(self.train_set))

    def evaluate(self):
        accumulative_acc = 0
        for sent in self.test_set:
            correct_arcs = 0
            mst = min_spanning_arborescence_nx(self.get_arcs(sent, self.weights), None)
            for node in sent.nodes:
                checked_word = sent.nodes[node]['word']
                if checked_word == self.root_word:
                    continue
                true_head = sent.nodes[node]['head']
                predicted_head = mst[checked_word].head
                if true_head == predicted_head:
                    correct_arcs += 1
            accumulative_acc += correct_arcs / (len(sent.nodes) - 1) # the -1 is for not counting the root which is not really a word in the sentence

        return accumulative_acc / len(self.test_set)



def main():
    nltk.download('dependency_treebank')
    sentences = dependency_treebank.parsed_sents()
    mst_parser = MSTParser(sentences) # indludes model training
    print(f"Accuracy for the model is: {mst_parser.evaluate()}")



if __name__ == "__main__":
    main()
