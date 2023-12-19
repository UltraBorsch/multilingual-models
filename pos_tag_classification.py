import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
print(f"Pytorch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, hidden=None):
        embeds = self.word_embeddings(sentence)
        lstm_out, hidden = self.lstm(embeds.view(len(sentence), 1, -1), hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.log_softmax(tag_space, dim=1)
        return tag_scores, hidden


def preprocess_data(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.strip().split('\t')
            if len(parts) > 3:
                word, pos_tag = parts[1], parts[3]
                if pos_tag not in ['PUNCT', 'SYM', 'X']:
                    current_sentence.append((word, pos_tag))

    if current_sentence:  # Add the last sentence if file doesn't end with a comment
        sentences.append(current_sentence)

    return sentences

def build_vocab_index(sentences):
    word_to_ix = {"<UNK>": 0}  # Add an unknown token
    for sentence in sentences:
        for word, _ in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def build_tag_index(sentences):
    tag_to_ix = {}
    for sentence in sentences:
        for _, tag in sentence:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return tag_to_ix

def prepare_sequences(sentences, word_to_ix, tag_to_ix):
    sequences = []
    for sentence in sentences:
        idxs = [word_to_ix.get(word, word_to_ix["<UNK>"]) for word, _ in sentence]  # Use get() with default
        tags = [tag_to_ix[tag] for _, tag in sentence]
        sequences.append((torch.tensor(idxs, dtype=torch.long), torch.tensor(tags, dtype=torch.long)))
    return sequences

def evaluate_model(model, test_data):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for sentence, tags in test_data:
            sentence, tags = sentence.to(device), tags.to(device)
            tag_scores, _ = model(sentence)
            predicted_tags = torch.argmax(tag_scores, dim=1)

            all_predictions.extend(predicted_tags.tolist())
            all_true_labels.extend(tags.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted', zero_division=0.0)
    accuracy = sum([1 for true, pred in zip(all_true_labels, all_predictions) if true == pred]) / len(all_true_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


DATAPATH = "data/lstm/" # location of data folder relative to the program
EPOCHS = 30 # number of epochs to run each model for
LIMITDATA = True # whether or not to limit all datasets to the size of the first one. note does not effect evaluation
LANGUAGES = ["en", "fr", "es", "de", "ru", "zh", "ja", "ko", "fa", "ar"] # languages considered
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

first_language = True
train_size = 0;
valid_size = 0;

for language in LANGUAGES:
    print(f"Modelling {language}")

    file_path = DATAPATH + language + '/' + language + '-ud-train.conllu'
    sentences = preprocess_data(file_path)

    file_path = DATAPATH + language + '/' + language + '-ud-dev.conllu'
    valid_sentences = preprocess_data(file_path)

    # Step 1: Data Preparation
    # Assuming 'sentences' contains the preprocessed training data
    word_to_ix = build_vocab_index(sentences)
    tag_to_ix = build_tag_index(sentences)

    # Convert sentences to indices and pad them
    train_data = prepare_sequences(sentences, word_to_ix, tag_to_ix)
    valid_data = prepare_sequences(valid_sentences, word_to_ix, tag_to_ix)

    if first_language:
        train_size = len(train_data)
        valid_size = len(valid_data)
        first_language = False

    # Step 2: Model Definition
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)).to(device)

    # Step 3: Training the Model
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.08, weight_decay=0.0025, momentum=0.9, nesterov=True) # SGD learns slower but more correct iirc
    optimizer = optim.Adam(model.parameters(), lr=0.018, weight_decay=0.0008, amsgrad=True) # Adam learns faster but more wrong iirc

    hidden = None  # Reset hidden state at the beginning of each epoch

    min_valid_loss = np.inf

    for epoch in range(EPOCHS):
        train_loop = tqdm(train_data[:min(len(train_data), train_size)], desc=f"Epoch {epoch+1}/{EPOCHS}")

        model.train() #set model to training mode

        train_loss = 0.0
        for sentence, tags in train_loop:
            sentence, tags = sentence.to(device), tags.to(device)

            # Detach hidden state
            hidden = tuple(h.detach() for h in hidden) if hidden else None

            model.zero_grad()
            tag_scores, hidden = model(sentence, hidden)
            loss = loss_function(tag_scores, tags)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item(), lossSum=train_loss/len(train_loop))

        hidden = None  # Reset hidden state
        valid_loop = tqdm(valid_data[:min(len(valid_data), valid_size)], desc="Validation phase: ")
        valid_loss = 0.0
        model.eval()
        for sentence, tags in valid_loop:
            sentence, tags = sentence.to(device), tags.to(device)

            tag_scores, hidden = model(sentence, hidden)
            loss = loss_function(tag_scores, tags)
            valid_loss += loss.item() 

            valid_loop.set_postfix(ValidationLossSum=valid_loss/len(valid_loop))

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')

    model.load_state_dict(torch.load('saved_model.pth'))
    test_file_path = DATAPATH + language + '/' + language + '-ud-test.conllu'  # Make sure this path is correct
    test_sentences = preprocess_data(test_file_path)
    test_data = prepare_sequences(test_sentences, word_to_ix, tag_to_ix)
    evaluate_model(model, test_data)
