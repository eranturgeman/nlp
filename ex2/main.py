import nltk
import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from collections import defaultdict

TEST_SET_PROPORTION = 0.1
PROB_FOR_UNKNOWN = 1
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
    def __init__(self, train_set, test_set, add_one_smoothing):
        self.train_set = train_set
        self.test_set = test_set
        self.emission = defaultdict(default)
        self.categories_count = defaultdict(int)
        self.categories_pairs_count = defaultdict(default)
        self.transition = defaultdict(default)
        self.tags_words_count = defaultdict(default)
        self.all_words = set()
        self.add_one_smoothing = add_one_smoothing

    def fit(self):
        self.fit_emission()
        self.fit_transition()

    def fit_emission(self):
        for sentences in self.train_set:
            for word, tag in sentences:
                self.tags_words_count[tag][word] += 1
                self.all_words.add(word)

        for tag, words in self.tags_words_count.items():
            for word in words:
                if self.add_one_smoothing:
                    self.emission[str(tag)][str(word)] = self.calc_smoothed_emission(word, tag)
                else:
                    self.emission[str(tag)][str(word)] = self.calc_emission(word, tag)

    def calc_emission(self, word, tag):
        # if word not in self.tags_words_count[tag]:
        #     return 0 # # if tag == UNKNOWN_TAG else 0 #todo check what should be returned for unseen word

        total_tag_count = sum(count for count in self.tags_words_count[tag].values())
        return self.tags_words_count[tag][word] / total_tag_count

    def calc_smoothed_emission(self, word, tag):
        total_tag_count = sum(count for count in self.tags_words_count[tag].values())
        return (self.tags_words_count[tag][word] + 1) / (total_tag_count + len(self.all_words))

    def fit_transition(self):
        for sentence in self.train_set:
            prev = STARTING_TAG
            for word, tag in sentence:
                self.categories_count[prev] += 1
                self.categories_pairs_count[prev][tag] += 1
                prev = tag
            self.categories_count[prev] += 1
            self.categories_pairs_count[prev][END_TAG] += 1

        for first_tag, second_tags in self.categories_pairs_count.items():
            for second_tag in second_tags.keys():
                self.transition[first_tag][second_tag] = self.categories_pairs_count[first_tag][second_tag] / self.categories_count[first_tag]

        import json

        # print(json.dumps(self.transition, indent=2))
        pass

    def viterbi(self, sentence):
        categories = list(self.categories_count.keys())
        categories.remove('*')
        num_categories = len(categories)
        n = len(sentence)

        backpointers = np.zeros((n, num_categories))
        pi = np.zeros((n, num_categories))
        for k, (word, _) in enumerate(sentence):
            for cur_category_idx, cur_category in enumerate(categories):
                if word not in self.all_words and cur_category == 'NN':
                    e = 1
                else:
                    e = self.emission[cur_category][word]

                if k == 0:
                    pi[k, cur_category_idx] = 1 * self.transition[STARTING_TAG][cur_category] * e
                else:
                    max_calc = -float('inf')
                    max_category_idx = None
                    for prev_category_idx, prev_category in enumerate(categories):
                        prob_cur_category = pi[k-1, prev_category_idx] * self.transition[prev_category][cur_category] * e
                        if prob_cur_category > max_calc:
                            max_calc = prob_cur_category
                            max_category_idx = prev_category_idx

                    pi[k, cur_category_idx] = max_calc
                    backpointers[k, cur_category_idx] = max_category_idx

        # Find the last category
        max_cat_idx = None
        max_cat_prob = -float('inf')

        for cat_idx, cat_val in enumerate(pi[n-1]):
            p = cat_val * self.transition[categories[cat_idx]][END_TAG]
            if p > max_cat_prob:
                max_cat_idx = cat_idx
                max_cat_prob = p

        predicted_categories = [categories[max_cat_idx]]
        backpointers = backpointers.astype(int)
        for idx in range(n-1, 0, -1):
            prev_cat_idx = backpointers[idx, max_cat_idx]
            predicted_categories.append(categories[prev_cat_idx])
            max_cat_idx = prev_cat_idx

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
                # print(true_tag, predicted_tag)
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
    train_set, test_set = train_test_split(dataset, test_size=TEST_SET_PROPORTION, shuffle=False)

    # Question B: basic model
    model = BasicModel(train_set, test_set)
    model.fit()
    total_e, known_e, unknown_e = model.compute_error_rate()
    print("B. MLE error rate:")
    print(f"total error rate: {total_e}")
    print(f"known words error rate: {known_e}")
    print(f"unknown words error rate: {unknown_e}")

    # # todo fix
    # #Question C: BigramHMM
    bigram = BigramHMM(train_set, test_set, add_one_smoothing=False)
    bigram.fit()
    total_e, known_e, unknown_e = bigram.compute_error_rate()
    print('\nC. Bigram HMM tagger error rate')
    print(f"total error rate: {total_e}")
    print(f"known words error rate: {known_e}")
    print(f"unknown words error rate: {unknown_e}")

    #Question D: BigramHMM with add-one smoothing
    smoothed_bigram = BigramHMM(train_set, test_set, add_one_smoothing=True)
    smoothed_bigram.fit()
    total_e, known_e, unknown_e = smoothed_bigram.compute_error_rate()
    print('\nD. Bigram HMM tagger with add-one smoothing error rate')
    print(f"total error rate: {total_e}")
    print(f"known words error rate: {known_e}")
    print(f"unknown words error rate: {unknown_e}")