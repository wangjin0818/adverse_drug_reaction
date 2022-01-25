import os
import sys
import logging
import pickle
import ast
import pandas as pd
from collections import defaultdict

def load_train_file(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []; labels = []; label_tag_dict = {}; tag_label_dict = {}
    temp_dict = defaultdict(float)
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)
        texts.append(text)

        label = ast.literal_eval(train_df['labels'][i])

        for j in range(len(label)):
            temp_dict[label[j]] += 1

    for i, label in enumerate(temp_dict.keys()):
        label_tag_dict[label] = i
        tag_label_dict[i] = label

    for i, text in enumerate(train_df['norm_text']):
        label = ast.literal_eval(train_df['labels'][i])
        line_label = []

        for j in range(len(label)):
           # line_label.append(label_tag_dict[label[j]])
           line_label.append(label[j])

        labels.append(line_label)

    print("text length: " + str(len(texts)))
    print("label length: " + str(len(labels)))
    print(temp_dict)
    print(label_tag_dict)
    print(tag_label_dict)

    return texts, labels, label_tag_dict, tag_label_dict


def load_train_file11(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = [];
    labels = [];
    label_tag_dict = {};
    tag_label_dict = {}
    temp_dict = defaultdict(float)
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)

        texts.append(text)

        label = ast.literal_eval(train_df['labels'][i])

        for j in range(len(label)):
            temp_dict[label[j]] += 1

    for i, label in enumerate(temp_dict.keys()):
        label_tag_dict[label] = i
        tag_label_dict[i] = label

    for i, text in enumerate(train_df['norm_text']):
        label = ast.literal_eval(train_df['labels'][i])
        line_label = []

        for j in range(len(label)):
            # line_label.append(label_tag_dict[label[j]])
            line_label.append(label[j])
        labels.append(line_label)

    print("text length: " + str(len(texts)))
    print("label length: " + str(len(labels)))
    print(temp_dict)
    print(label_tag_dict)
    print(tag_label_dict)

    return texts, labels, label_tag_dict, tag_label_dict


def load_test_file(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)
        texts.append(text)

    return texts

asu_train_file = os.path.join('corpus', 'train', 'asu_train')
asu_test_file = os.path.join('corpus', 'test', 'asu_test')

asu_train_texts, asu_train_labels, label_tag_dict, tag_label_dict = load_train_file(asu_train_file)
asu_test_texts, asu_test_labels, _, _ = load_train_file(asu_test_file)


chop_train_file = os.path.join('corpus', 'train', 'chop_train')
chop_test_file = os.path.join('corpus', 'test', 'chop_test')

chop_train_texts, chop_train_labels, _, _ = load_train_file11(chop_train_file)
chop_test_texts, chop_test_labels, _, _ = load_train_file11(chop_test_file)

train_texts, train_labels = [], []
test_texts, test_labels = [], []

for i, text in enumerate(asu_train_texts):
    train_texts.append(text)
    train_labels.append(asu_train_labels[i])

for i, text in enumerate(asu_test_texts):
    test_texts.append(text)
    test_labels.append(asu_test_labels[i])

for i, text in enumerate(chop_train_texts):
    train_texts.append(text)
    train_labels.append(chop_train_labels[i])

for i, text in enumerate(chop_test_texts):
    test_texts.append(text)
    test_labels.append(chop_test_labels[i])


pickle_file = "./pickle/twitter_IO.pickle3"
pickle.dump([train_texts, train_labels, test_texts, test_labels, label_tag_dict, tag_label_dict],
            open(pickle_file, 'wb'))

train_texts, train_labels, test_texts, test_labels, label_tag_dict1, tag_label_dict1 = pickle.load(
    open(pickle_file, 'rb'))
