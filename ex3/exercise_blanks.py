import torch #todo check on torch 1.3 version
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
LSTM_EMBEDDING_DIM = 300
LOG_LINEAR_EPOCH_NUM = 20
LOG_LINEAR_LEARNING_RATE = 0.01
LOG_LINEAR_BATCH_SIZE = 64
LOG_LINEAR_WIGHT_DECAY = 0.001
LSTM_DIM = 100
LSTM_LEARNING_RATE = 0.001
LSTM_WIGHT_DECAY = 0.0001
LSTM_DROPOUT = 0.5
LSTM_BATCH_SIZE = 64
LSTM_EPOCH_NUM = 4


ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"
LOG_LINEAR_PATH = "log_linear_models/log_linear.model.epoch"
W2V_PATH = "w2v_models/w2v.model.epoch"
LSTM_PATH = "lstm_models/lstm.model.epoch"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE_WORDS = 'rare'
NEGATED_POLARITY = "negated polarity"

LOAD_MODEL = False


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys()) #todo change vocab to key_to_index ?
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    res = np.zeros(embedding_dim)
    leaves = sent.get_leaves()
    known_words_amount = 0

    for leave in leaves:
        assert len(leave.text) == 1  # todo DEL
        word = leave.text[0]
        if word in word_to_vec:  #todo check if should sum the zero-vecs for unknown words (and increase denomenator)
            res += word_to_vec[word]
            known_words_amount += 1

    return (res / known_words_amount) if known_words_amount != 0 else res


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    res = np.zeros(size)
    res[ind] = 1
    return res


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """

    # vocabulary_size = len(word_to_ind)
    # res = np.zeros(vocabulary_size)
    # leaves = sent.get_leaves()
    # for leave in leaves:
    #     assert len(leave.text) == 1 #todo DEL
    #     word = leave.text[0]
    #     res += get_one_hot(vocabulary_size, word_to_ind[word])
    #
    # return res / len(leaves)

    vocabulary_size = len(word_to_ind)
    res = np.zeros(vocabulary_size)
    leaves = sent.get_leaves()
    for leave in leaves:
        assert len(leave.text) == 1  # todo DEL
        res[word_to_ind[leave.text[0]]] = 1
    return res / len(leaves)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: i for i, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    res = []
    leaves = sent.get_leaves()
    words_amount = 0

    while words_amount < seq_len and words_amount < len(leaves):
        word = leaves[words_amount]
        if word in word_to_vec:
            word_vec = word_to_vec[word]
        else:
            word_vec = np.zeros(embedding_dim)
        res.append(word_vec)
        words_amount += 1

    if words_amount < seq_len:
        for i in range(seq_len - words_amount):
            res.append(np.zeros(embedding_dim))

    return np.array(res)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        rare_words_indices = data_loader.get_rare_words_examples(self.sentiment_dataset.get_test_set(), self.sentiment_dataset)
        self.sentences[RARE_WORDS] = [sentence for i, sentence in enumerate(self.sentiment_dataset.get_test_set()) if i in rare_words_indices]

        negated_polarity_indices = data_loader.get_negated_polarity_examples(self.sentiment_dataset.get_test_set())
        self.sentences[NEGATED_POLARITY] = [sentence for i, sentence in enumerate(self.sentiment_dataset.get_test_set()) if i in negated_polarity_indices]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape

# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden2tag_layer = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        output, (h_n, c_n) = self.lstm_layer(text)
        hidden_layer_input = self.dropout_layer(h_n)
        return self.hidden2tag_layer(hidden_layer_input)

    def predict(self, text):
        return torch.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # DO NOT implement sigmoid here
        return self.linear(x)

    def predict(self, x):
        # here we use sigmoid since we will not use cross entropy on test set and we want the prediction right away
        return torch.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # assume preds are already rounded to 0 or 1

    number_of_examples = y.shape[0]
    #todo check what should 0.5 be rounded to- up or down?
    y = np.around(y)
    preds = np.around(preds)
    return ((y == preds).sum()) / number_of_examples


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    for embedding_batch, label_batch in data_iterator:
        forward_res = model.forward(embedding_batch.float()).reshape(-1)
        optimizer.zero_grad()
        loss_score = criterion(forward_res, label_batch) #todo make sure in the criterion we activate sigmoid
        loss_score.backward()
        optimizer.step()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    avg_loss_arr = []
    avg_accuracy_arr = []

    with torch.no_grad():
        for embedding_batch, label_batch in data_iterator:
            forward_res = model.forward(embedding_batch.float()).reshape(-1)
            loss_score = criterion(forward_res, label_batch) #todo make sure in the criterion we activate sigmoid
            avg_loss_arr.append(loss_score.item())
            avg_accuracy_arr.append(binary_accuracy(forward_res, label_batch))

    return np.mean(avg_loss_arr), np.mean(avg_accuracy_arr)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    res = []
    for embedding_batch, _ in data_iter:
        res.extend(model.predict(embedding_batch))

    return np.array(res)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0., model_path=LOG_LINEAR_PATH):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    :param model_path: model path
    """
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        print(f"entering epoch number {epoch}")
        final_path = model_path + f"{epoch}"
        #if not LOAD_MODEL: todo del
        if not os.path.exists(final_path):
            print(f"model number {epoch} trained") #todo del
            train_epoch(model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion)
            save_model(model, final_path, epoch, optimizer)
        else:
            model, optimizer, epoch = load(model, final_path, optimizer)

        #train set eval
        loss, acc = evaluate(model, data_manager.get_torch_iterator(TRAIN), criterion)
        train_loss.append(loss)
        train_acc.append(acc)

        #validation set eval
        loss, acc = evaluate(model, data_manager.get_torch_iterator(VAL), criterion)
        validation_loss.append(loss)
        validation_acc.append(acc)

    return train_loss, train_acc, validation_loss, validation_acc


def plot_log_linear_graphs(x_axis, y_axis1, y_axis2, x_label="", y_label="", title="", png_name="plot.png"):
    plt.plot(x_axis, y_axis1, 'r', label="train data")
    plt.plot(x_axis, y_axis2, 'b', label="validation data")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(png_name)
    plt.show()
    plt.clf()


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    device = get_available_device()
    data_manager = DataManager(batch_size=LOG_LINEAR_BATCH_SIZE)
    log_linear = LogLinear(data_manager.get_input_shape()[0]).to(device)

    train_loss, train_acc, validation_loss, validation_acc = \
        train_model(log_linear, data_manager, LOG_LINEAR_EPOCH_NUM, LOG_LINEAR_LEARNING_RATE, LOG_LINEAR_WIGHT_DECAY)

    epochs = [i for i in range(LOG_LINEAR_EPOCH_NUM)]
    plot_log_linear_graphs(epochs, train_loss, validation_loss,
                           "Number of epochs", "Loss", "Train & Validation Loss", "train_validation_loss.png")

    plot_log_linear_graphs(epochs, train_acc, validation_acc,
                               "Number of epochs", "Accuracy", "Train & Validation Accuracy", "train_validation_acc.png")

    # results for test set and special subsets
    criterion = nn.BCEWithLogitsLoss()
    data_iterator = data_manager.get_torch_iterator(TEST)
    test_loss, test_acc = evaluate(log_linear, data_iterator, criterion)
    print(f"LOG LINEAR | test loss: {test_loss}")
    print(f"LOG LINEAR | test accuracy: {test_acc}")

    rare_iterator = data_manager.get_torch_iterator(RARE_WORDS)
    rare_loss, rare_acc = evaluate(log_linear, rare_iterator, criterion)
    print(f"LOG LINEAR | test rare words loss: {rare_loss}")
    print(f"LOG LINEAR | test rare words accuracy: {rare_acc}")

    negated_iterator = data_manager.get_torch_iterator(NEGATED_POLARITY)
    negated_loss, negated_acc = evaluate(log_linear, negated_iterator, criterion)
    print(f"LOG LINEAR | test negated polarity loss: {negated_loss}")
    print(f"LOG LINEAR | test negated polarity accuracy: {negated_acc}")

    #todo del those lines

    # print(f"train loss:\n {train_loss}")
    # print(f"train accuracy:\n {train_acc}")
    # print(f"validation loss:\n {validation_loss}")
    # print(f"validation accuracy:\n {validation_acc}")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    device = get_available_device()
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=LOG_LINEAR_BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    w2v_model = LogLinear(data_manager.get_input_shape()[0]).to(device)

    train_loss, train_acc, validation_loss, validation_acc = \
        train_model(w2v_model, data_manager, LOG_LINEAR_EPOCH_NUM, LOG_LINEAR_LEARNING_RATE, LOG_LINEAR_WIGHT_DECAY, model_path=W2V_PATH)

    epochs = [i for i in range(LOG_LINEAR_EPOCH_NUM)]
    plot_log_linear_graphs(epochs, train_loss, validation_loss,
                           "Number of epochs", "Loss", "Train & Validation Loss", "LL_train_validation_loss.png")

    plot_log_linear_graphs(epochs, train_acc, validation_acc,
                           "Number of epochs", "Accuracy", "Train & Validation Accuracy", "LL_train_validation_acc.png")

    # results for test set and special subsets
    criterion = nn.BCEWithLogitsLoss()
    data_iterator = data_manager.get_torch_iterator(TEST)
    test_loss, test_acc = evaluate(w2v_model, data_iterator, criterion)
    print(f"W2V MODEL | test loss: {test_loss}")
    print(f"W2V MODEL | test accuracy: {test_acc}")

    rare_iterator = data_manager.get_torch_iterator(RARE_WORDS)
    rare_loss, rare_acc = evaluate(w2v_model, rare_iterator, criterion)
    print(f"W2V MODEL | test rare words loss: {rare_loss}")
    print(f"W2V MODEL | test rare words accuracy: {rare_acc}")

    negated_iterator = data_manager.get_torch_iterator(NEGATED_POLARITY)
    negated_loss, negated_acc = evaluate(w2v_model, negated_iterator, criterion)
    print(f"W2V MODEL | test negated polarity loss: {negated_loss}")
    print(f"W2V MODEL | test negated polarity accuracy: {negated_acc}")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    device = get_available_device()
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=LSTM_BATCH_SIZE, embedding_dim=LSTM_EMBEDDING_DIM)
    lstm_model = LSTM(data_manager.get_input_shape()[0], LSTM_DIM, 1, LSTM_DROPOUT).to(device)

    train_loss, train_acc, validation_loss, validation_acc = \
        train_model(lstm_model, data_manager, LSTM_EPOCH_NUM, LSTM_LEARNING_RATE, LSTM_WIGHT_DECAY, model_path=LSTM_PATH)

    epochs = [i for i in range(LSTM_EPOCH_NUM)]
    plot_log_linear_graphs(epochs, train_loss, validation_loss,
                           "Number of epochs", "Loss", "Train & Validation Loss", "LL_train_validation_loss.png")

    plot_log_linear_graphs(epochs, train_acc, validation_acc,
                           "Number of epochs", "Accuracy", "Train & Validation Accuracy", "LL_train_validation_acc.png")

    # results for test set and special subsets
    criterion = nn.BCEWithLogitsLoss()
    data_iterator = data_manager.get_torch_iterator(TEST)
    test_loss, test_acc = evaluate(lstm_model, data_iterator, criterion)
    print(f"W2V MODEL | test loss: {test_loss}")
    print(f"W2V MODEL | test accuracy: {test_acc}")

    rare_iterator = data_manager.get_torch_iterator(RARE_WORDS)
    rare_loss, rare_acc = evaluate(lstm_model, rare_iterator, criterion)
    print(f"W2V MODEL | test rare words loss: {rare_loss}")
    print(f"W2V MODEL | test rare words accuracy: {rare_acc}")

    negated_iterator = data_manager.get_torch_iterator(NEGATED_POLARITY)
    negated_loss, negated_acc = evaluate(lstm_model, negated_iterator, criterion)
    print(f"W2V MODEL | test negated polarity loss: {negated_loss}")
    print(f"W2V MODEL | test negated polarity accuracy: {negated_acc}")


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    train_lstm_with_w2v()