import nltk
import numpy as np
import pandas as pd
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re

TEST_SET_PROPORTION = 0.1
PROB_FOR_UNKNOWN = 1
UNKNOWN_TAG = "NN"
STARTING_TAG = "*"
END_TAG = "STOP"
PSEUDO_WORDS_THRESHOLD = 5

PSEUDO_WORDS_PATTERNS = {
    'twoDigitNum': '\d\d',
    'fourDigitNum': '\d\d\d\d',
    'containsDigitAndAlpha': '(.*[a-zA-Z]\d.*)|(.*\d[a-zA-Z].*)',
    'containsDigitAndDash': '\d+-\d+',
    'containsDateFormat': '(\d+/\d+/\d+)|(\d+.d+.\d+)',
    'containsDigitAndPeriod': '\d+\.\d+',
    'othernum': '\d+',
    'allCaps': '[A-Z]+',
    'capPeriod': '[A-Z].',
    'initCap': '[A-Z][a-z]+',
    'endsWithIng': '[A-Za-z]+ing',
    'endsWithS': '[A-Za-z]+s',
    'endsWithApostropheS': "[A-Za-z]+'s",
    'endsWithEd': "[A-Za-z]+ed",
    'lettersApostropheLetter': "[A-Za-z]+'[A-Za-z]",
    'lowercase': '[a-z]+',
}
OTHER_WORDS = 'OtherCategory'

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

        print(f"total error rate: {1 - total_words_accuracy}")
        print(f"known words error rate: {1 - known_words_accuracy}")
        print(f"unknown words error rate: {1 - unknown_words_accuracy}")


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
        self.number_of_words = 0

    def fit(self):
        self.fit_emission()
        self.fit_transition()

    def fit_emission(self):
        for sentences in self.train_set:
            for word, tag in sentences:
                self.tags_words_count[tag][word] += 1
                self.all_words.add(word)

        self.number_of_words = len(self.all_words)

        for tag, words in self.tags_words_count.items():
            total_tag_count = sum(self.tags_words_count[tag].values())

            for word in words:
                if self.add_one_smoothing:
                    self.emission[tag][word] = (self.tags_words_count[tag][word] + 1) / (total_tag_count + self.number_of_words)
                else:
                    self.emission[tag][word] = self.tags_words_count[tag][word] / total_tag_count

    def get_emission(self, word, category):
        if word not in self.all_words:
            if self.add_one_smoothing:
                # unknown word with smoothing
                total_tag_count = sum(self.tags_words_count[category].values())
                e = 1 / (total_tag_count + self.number_of_words)
            elif category == 'NN':
                # unknown word without smoothing with NN tag todo check if 1 should be returned
                e = 1
            else:
                # unknown word without smoothing without NN tag
                e = 0
        else:
            e = self.emission[category][word]

        return e

    def fit_transition(self):
        for sentence in self.train_set:
            prev = STARTING_TAG
            for word, tag in sentence:
                self.categories_count[prev] += 1
                self.categories_pairs_count[prev][tag] += 1
                prev = tag
            self.categories_count[prev] += 1
            self.categories_pairs_count[prev][END_TAG] += 1

        for first_tag, second_tags_dict in self.categories_pairs_count.items():
            for second_tag in second_tags_dict.keys():
                self.transition[first_tag][second_tag] = self.categories_pairs_count[first_tag][second_tag] / self.categories_count[first_tag]

    def viterbi(self, sentence):
        categories = list(self.categories_count.keys())
        categories.remove('*')
        num_categories = len(categories)
        n = len(sentence)

        pi = np.zeros((n, num_categories))
        backpointers = np.zeros((n, num_categories))

        for k, (word, _) in enumerate(sentence):
            for cur_category_idx, cur_category in enumerate(categories):
                e = self.get_emission(word, cur_category)

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

        return self.get_category_path(backpointers, categories, n, pi)

    def get_category_path(self, backpointers, categories, n, pi):
        # Find the last category
        max_cat_idx = None
        max_cat_prob = -float('inf')
        for cat_idx, cat_val in enumerate(pi[n - 1]):
            p = cat_val * self.transition[categories[cat_idx]][END_TAG]
            if p > max_cat_prob:
                max_cat_idx = cat_idx
                max_cat_prob = p
        predicted_categories = [categories[max_cat_idx]]
        backpointers = backpointers.astype(int)

        for idx in range(n - 1, 0, -1):
            prev_cat_idx = backpointers[idx, max_cat_idx]
            predicted_categories.append(categories[prev_cat_idx])
            max_cat_idx = prev_cat_idx

        return predicted_categories[::-1]

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

        print(f"total error rate: {1 - total_words_accuracy}")
        print(f"known words error rate: {1 - known_words_accuracy}")
        print(f"unknown words error rate: {1 - unknown_words_accuracy}")

    def get_confusion_matrix(self):
        print(self.categories_count)
        categories = set(self.categories_count.keys())
        for s in self.test_set:
            for w, t in s:
                categories.add(t)

        m = pd.DataFrame(columns=categories, index=categories).fillna(0)

        for sentence in self.test_set:
            predicted_tags = self.viterbi(sentence)

            for (word, true_tag), predicted_tag in zip(sentence, predicted_tags):
                m[predicted_tag][true_tag] += 1

        print(m)
        m.to_excel("output.xlsx")


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

def get_pseudo_word(word):
    for tag, pattern in PSEUDO_WORDS_PATTERNS.items():
        if re.fullmatch(pattern, word):
            return tag
    return OTHER_WORDS

def pseudo_words_data_set(dataset):
    words_counter = defaultdict(int)
    for sentence in dataset:
        for word, tag in sentence:
            words_counter[word] += 1

    for sentence in dataset:
        for i, (word, tag) in enumerate(sentence):
            if words_counter[word] < PSEUDO_WORDS_THRESHOLD:
                # Low Frequency words
                new_word = get_pseudo_word(word)
                sentence[i] = (new_word, tag)

    return dataset


if __name__ == '__main__':
    # reading, cleaning and splitting corpus
    nltk.download('brown')
    dataset = brown.tagged_sents(categories='news')
    dataset = clean_dataset(dataset)
    train_set, test_set = train_test_split(dataset, test_size=TEST_SET_PROPORTION, shuffle=False)
    
    # Question B: basic model
    print("B. MLE error rate:")
    model = BasicModel(train_set, test_set)
    model.fit()
    model.compute_error_rate()
    
    #Question C: BigramHMM
    print('\nC. Bigram HMM tagger error rate')
    bigram = BigramHMM(train_set, test_set, add_one_smoothing=False)
    bigram.fit()
    bigram.compute_error_rate()
    
    #Question D: BigramHMM with add-one smoothing
    print('\nD. Bigram HMM tagger with add-one smoothing error rate')
    smoothed_bigram = BigramHMM(train_set, test_set, add_one_smoothing=True)
    smoothed_bigram.fit()
    smoothed_bigram.compute_error_rate()
    
    print('\nE.ii Bigram HMM tagger with PseudoWords')
    smoothed_bigram = BigramHMM(pseudo_words_data_set(list(train_set)), pseudo_words_data_set(list(test_set)), add_one_smoothing=False)
    smoothed_bigram.fit()
    smoothed_bigram.compute_error_rate()

    print('\nE.iii Bigram HMM tagger with PseudoWords with Add-One smoothing')
    smoothed_bigram = BigramHMM(pseudo_words_data_set(list(train_set)), pseudo_words_data_set(list(test_set)), add_one_smoothing=True)
    smoothed_bigram.fit()
    smoothed_bigram.compute_error_rate()
    smoothed_bigram.get_confusion_matrix()
