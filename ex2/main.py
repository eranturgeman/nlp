import nltk
import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from collections import defaultdict

TEST_SET_PROPORTION = 0.1
PROB_FOR_UNKNOWN = -float("inf")
UNKNOWN_TAG = "NN"
STARTING_TAG = "*"
END_TAG = "STOP"
VITERBI_UNKNOWN = "VUK"

def default():
    return defaultdict(int)


class BasicModel:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.word_tags = defaultdict(default)

    def fit(self):
        for sentence in self.train_set:
            for word, tag in sentence:
                self.word_tags[word][tag] += 1

    def predict_best_tag_probability(self, word):
        if word not in self.word_tags:
            return PROB_FOR_UNKNOWN, UNKNOWN_TAG # todo check if leaving -inf in unknown probabilities

        categories = self.word_tags[word]

        max_count = -float("inf")
        max_category = None
        total_count = 0
        for category, count in categories.items():
            total_count += count
            if count > max_count:
                max_count = count
                max_category = category

        return max_count / total_count, max_category

    def compute_error_rate(self):
        total_words = 0
        correct_tag_words = 0
        known_words = 0
        known_words_correct_tag = 0
        unknown_words = 0
        unknown_words_correct_tag = 0

        for sentence in self.test_set:
            for word, tag in sentence:
                total_words += 1

                if word in self.word_tags:
                    known_words += 1
                else:
                    unknown_words += 1

                if self.predict_best_tag_probability(word)[1] == tag:
                    correct_tag_words += 1

                    if word in self.word_tags:
                        known_words_correct_tag += 1
                    else:
                        unknown_words_correct_tag += 1

        total_words_accuracy = correct_tag_words / total_words
        known_words_accuracy = known_words_correct_tag / known_words
        unknown_words_accuracy = unknown_words_correct_tag / unknown_words
        return 1 - total_words_accuracy, 1 - known_words_accuracy, 1 - unknown_words_accuracy #total error, know error, unknown error


class BigramHMM:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.emission = defaultdict(default)
        self.categories_count = dict()
        self.categories_pairs_count = dict()
        self.transition = defaultdict(default)
        self.tags_words_count = defaultdict(default)
        self.all_words = set()

    def fit(self):
        self.fit_emission()
        self.fit_transition()

    def fit_emission(self):
        for sentences in self.train_set:
            for word, tag in sentences:
                self.tags_words_count[tag][word] += 1
                self.all_words.add(word)

    def fit_transition(self):
        prev = STARTING_TAG
        for sentence in self.train_set:
            for word, tag in sentence:
                self.categories_count[prev] += 1
                self.categories_pairs_count[prev][tag] += 1
                prev = tag
            self.categories_count[prev] += 1
            self.categories_pairs_count[prev][END_TAG] += 1

    def predict_emission(self, word, tag):
        if word not in self.tags_words_count[tag]:
            return 0 #todo check what should be returned for unseen word

        total_tag_count = sum(count for _, count in self.tags_words_count[tag])
        return self.tags_words_count[tag][word] / total_tag_count

    def predict_transition(self, first_tag, second_tag):
        return self.categories_pairs_count[first_tag][second_tag] / self.categories_count[first_tag]

    def viterbi(self, sentence):
        categories = list(self.categories_count.keys())
        num_categories = len(categories)
        n = len(sentence)

        prev_pi = None

        backpointers = np.zeros((n, num_categories))

        for k, word in enumerate(sentence):
            pi = np.ones(num_categories)

            for cur_category_idx, cur_category in enumerate(categories):
                if k == 1:
                    pi[cur_category_idx] = self.predict_transition(STARTING_TAG, cur_category) *\
                                           self.predict_emission(word, cur_category)
                else:
                    max_calc = -float("inf")
                    max_category_idx = None
                    for prev_category_idx, prev_category in enumerate(categories):
                        prob_cur_category = prev_pi[prev_category_idx] *\
                                            self.predict_transition(prev_category, cur_category) *\
                                            self.predict_emission(word, cur_category)
                        if prob_cur_category > max_calc:
                            max_calc = prob_cur_category
                            max_category_idx = prev_category_idx

                    pi[cur_category_idx] = max_calc
                    backpointers[k, cur_category_idx] = max_category_idx
            prev_pi = pi

        # Find the last category
        max_cat_idx = None
        max_cat_prob = -float('inf')

        for cat_idx, cat_val in enumerate(prev_pi):
            p = cat_val * self.predict_transition(categories[cat_idx], END_TAG)
            if p > max_cat_prob:
                max_cat_idx = cat_idx
                max_cat_prob = p

        predicted_categories = [categories[max_cat_idx]]
        for idx in range(n-2, 0, -1):
            prev_cat_idx = backpointers[idx, max_cat_idx]
            predicted_categories.append(categories[prev_cat_idx])

        return predicted_categories[::-1]

        #for the k-th word in sentence:
        #   for cur_cat in categories:
        #       if k == 1:
        #           ==== pi(0, *) = 1 ====
        #           x_k = pi(k - 1, *) * q(cur_cat | *) * e(word | cur_cat)
        #           pi(k, cur_cat) = x_k
        #       else:
        #           for prev_cat in categories:
        #                x_k = pi(k - 1, prev_cat) * q(cur_cat | prev_cat) * e(word | cur_cat)
        #           pi(k, cur_cat) = argmax{ x_k from prev_cat }

    def compute_error_rate(self):
        total_words = 0
        correct_tag_words = 0
        known_words = 0
        known_words_correct_tag = 0
        unknown_words = 0
        unknown_words_correct_tag = 0

        for sentence in self.test_set:
            predicted_tags = self.viterbi(sentence)

            for (word, true_tag), predicted_tag in zip(sentence, predicted_tags):
                total_words += 1

                if word in self.all_words:
                    known_words += 1
                else:
                    unknown_words += 1

                if predicted_tag == true_tag:
                    correct_tag_words += 1

                    if word in self.all_words:
                        known_words_correct_tag += 1
                    else:
                        unknown_words_correct_tag += 1

        total_words_accuracy = correct_tag_words / total_words
        known_words_accuracy = known_words_correct_tag / known_words
        unknown_words_accuracy = unknown_words_correct_tag / unknown_words
        return 1 - total_words_accuracy, 1 - known_words_accuracy, 1 - unknown_words_accuracy  # total error, know error, unknown error



# general functions
def filter_tag(tag):
    #todo check if * can appear as prefix. if so modify this func to handle it
    if '*' in tag:
        idx = tag.find("*")
        tag = tag[:idx]

    if '-' in tag:
        idx = tag.find("-")
        tag = tag[:idx]

    if '+' in tag:
        idx = tag.find("+")
        tag = tag[:idx]

    return tag

def clean_dataset(dataset):
    clean = []
    for sentence in dataset:
        clean_sentence = [(word, filter_tag(tag)) for word, tag in sentence]
        clean.append(clean_sentence)

    return clean

if __name__ == '__main__':
    # reading, cleaning and splitting corpus
    nltk.download('brown')
    dataset = brown.tagged_sents(categories='news')
    dataset = clean_dataset(dataset)
    train_set, test_set = train_test_split(dataset, test_size=TEST_SET_PROPORTION)

    # Question B: basic model
    model = BasicModel(train_set, test_set)
    model.fit()
    total_e, known_e, unknown_e = model.compute_error_rate()
    print(f"total error rate: {total_e}")
    print(f"known words error rate: {known_e}")
    print(f"unknown words error rate: {unknown_e}")

    #Question C: BigramHMM
    bigram = BigramHMM(train_set, test_set)
    bigram.fit()
    total_e, known_e, unknown_e = bigram.compute_error_rate()
    print(f"total error rate: {total_e}")
    print(f"known words error rate: {known_e}")
    print(f"unknown words error rate: {unknown_e}")