import nltk
from nltk.corpus import brown
# from sklearn.model_selection import train_test_split
from collections import defaultdict

TEST_SET_PROPORTION = 0.1
PROB_FOR_UNKNOWN = -float("inf")
UNKNOWN_TAG = "NN"

class PosModel:
    def __init__(self, sentences):
        self.sentences = sentences
        self.word_tags = defaultdict(self.default)
        self.max_probability = dict()

    def fit(self):
        for sentence in self.sentences:
            for word, tag in sentence:
                self.word_tags[word][tag] += 1

        for word in self.word_tags.keys():
            self.find_best_tag_probability(word)

    def find_best_tag_probability(self, word):
        if word not in self.max_probability:
            self.max_probability[word] = (PROB_FOR_UNKNOWN, UNKNOWN_TAG) # todo check if leaving -inf in unknown probabilities
            return self.max_probability

        categories = self.word_tags[word]

        max_count = -float("inf")
        max_category = None
        total_count = 0
        for category, count in categories.items():
            total_count += count
            if count > max_count:
                max_count = count
                max_category = category

        self.max_probability[word] = (max_count / total_count, max_category)

        return self.max_probability[word]


    @staticmethod
    def default():
        return defaultdict(int)


if __name__ == '__main__':
    nltk.download('brown')
    sentences = brown.tagged_sents(categories='news')
    # train_set, test_set = train_test_split(sentences, test_size=TEST_SET_PROPORTION)
    model = PosModel(sentences)
    model.fit()
    print(model.find_best_tag_probability("said"))
    print(model.find_best_tag_probability("moshe"))

    # print("hi")

