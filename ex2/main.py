import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

TEST_SET_PROPORTION = 0.1



if __name__ == '__main__':
    nltk.download('brown')
    sentences = brown.tagged_sents(categories='news')
    train_set, test_set = train_test_split(sentences, test_size=TEST_SET_PROPORTION)

