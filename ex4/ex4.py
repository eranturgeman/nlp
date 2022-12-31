import itertools
import math
import random
import collections
import numpy as np
import nltk
from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

Arc = collections.namedtuple('Arc', ['head', 'tail', 'weight', 'head_pos', 'tail_pos', 'distance'])
TRAIN_SET_SIZE = 0.9
EPOCH_NUM = 2
ROOT_WORD_LEN = 15
LEARNING_RATE = 1
ROOT_POS = "TOP"
BONUS = 0 #put 1 to use bonus feature, else put 0

class MSTParser:
    def __init__(self, sentences):
        sentences = sentences[:250]
        self.sentences = sentences
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
        #adding root and root POS
        words.add(self.root_word)
        pos_tags.add(ROOT_POS)

        for sent in self.sentences:
            for node in sent.nodes:
                #todo check
                if sent.nodes[node]['tag'] == ROOT_POS: #if we are at the ROOT node
                    continue
                words.add(sent.nodes[node]['word'])
                pos_tags.add(sent.nodes[node]['tag'])

        self.words_mapping = {(a, b): i for (i, (a, b)) in enumerate(itertools.product(words, repeat=2))}
        words_offset = len(self.words_mapping)
        self.pos_mapping = {(a, b): (i + words_offset) for (i, (a, b)) in enumerate(itertools.product(pos_tags, repeat=2))}

        self.feature_vec_size = words_offset + len(self.pos_mapping) + BONUS

    def get_arcs(self, sent, w):
        arcs = []
        n = len(sent.nodes)
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue

                word1 = sent.nodes[i]['word']
                word2 = sent.nodes[j]['word']
                word1_pos = sent.nodes[i]['tag']
                word2_pos = sent.nodes[j]['tag']

                if  word2_pos == ROOT_POS:
                    continue
                if word1_pos == ROOT_POS:
                    word1 = self.root_word

                #assuming all pairs of 2 words and 2 POS are in dicts (running on test set at the beginning)
                score = w[self.words_mapping[(word1, word2)]] + w[self.pos_mapping[(word1_pos, word2_pos)]]

                if BONUS:
                    dist = abs(i - j)
                    score += w[self.feature_vec_size - 1] * (1 / dist)
                    arcs.append(Arc(word1, word2, -score, word1_pos, word2_pos, dist))
                else:
                    arcs.append(Arc(word1, word2, -score, word1_pos, word2_pos, 0))


        return arcs

    def update_weights(self, w, predicted_tree, sent):
        #TODO make sure we need to sub golden - predicted and not predicted - golden
        for node in sent.nodes:
            word = sent.nodes[node]['word']
            pos = sent.nodes[node]['tag']
            parent_idx = sent.nodes[node]['head']
            if not parent_idx: #if im the root and have no parent
                continue
            parent_word = sent.nodes[parent_idx]['word']
            parent_pos = sent.nodes[parent_idx]['tag']
            w[self.words_mapping[(parent_word, word)]] += 1
            w[self.pos_mapping[(parent_pos, pos)]] += 1
            if BONUS:
                w[self.feature_vec_size - 1] += (1 / abs(parent_idx - node))

        for arc in predicted_tree.values():
            w[self.words_mapping[(arc.head, arc.tail)]] -= 1
            w[self.pos_mapping[(arc.head_pos, arc.tail_pos)]] -= 1
            if BONUS:
                w[self.feature_vec_size - 1] -= (1 / arc.distance)

    def train(self):
        #iterate NUM_EPOCHS
        #for each epoch iterate over all sentences
        #for each sentence iterate over all pairs of following words
        #calc score for each pair and create an arc for each pair: (word1, word2, score)
        #keep all arcs of a sentence and send them to Edmonds to get T' for this sentence
        #update wights add to w_sum
        #normalize w_sum

        w_sum = np.zeros(self.feature_vec_size)
        w = np.zeros(self.feature_vec_size)

        for r in range(EPOCH_NUM):
            random.shuffle(self.train_set)
            k = 0  #todo del
            for sent in self.train_set:
                print(f"epoch {r} sentence {k}") #todo del
                full_sent_graph = self.get_arcs(sent, w)
                mst = min_spanning_arborescence_nx(full_sent_graph, None)
                self.update_weights(w, mst, sent)
                w_sum += w
                k += 1 #todo del

        return w_sum / (EPOCH_NUM * len(self.train_set))

    def evaluate(self):
        accumulative_acc = 0
        for sent in self.test_set:
            correct_arcs = 0
            full_sent_graph = self.get_arcs(sent, self.weights)
            mst = min_spanning_arborescence_nx(full_sent_graph, None)
            for node in sent.nodes:
                parent_idx = sent.nodes[node]['head']
                if not parent_idx:
                    continue
                checked_word = sent.nodes[node]['word']
                true_head = sent.nodes[parent_idx]['word']
                predicted_head = mst[checked_word].head
                if true_head == predicted_head:
                    correct_arcs += 1
            accumulative_acc += (correct_arcs / (len(sent.nodes) - 1)) # the -1 is for not counting the root which is not really a word in the sentence

        return accumulative_acc / len(self.test_set)


def main():
    nltk.download('dependency_treebank')
    sentences = dependency_treebank.parsed_sents()
    mst_parser = MSTParser(sentences) # indludes model training
    print(f"Accuracy for the model is: {mst_parser.evaluate()}")



if __name__ == "__main__":
    main()
