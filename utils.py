import os
import sys
import logging

import ast
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def load_twitter_train(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []; labels = []; label_tag_dict = {}; tag_label_dict = {}
    for i, text in enumerate(train_df['text']):
        print(text)
        start = train_df['start'][i]
        end = train_df['end'][i]
        print(text[start:end])
        print()


train_file = os.path.join('data', 'twitter adr', 'processed', 'train', 'asu_train')
load_twitter_train(train_file)


# def load_twitter_train(filename):
#     train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')
#
#     texts = []; labels = []; label_tag_dict = {}; tag_label_dict = {}
#     temp_dict = defaultdict(float)
#     for i, text in enumerate(train_df['norm_text']):
#         # text = ast.literal_eval(text)
#         texts.append(text)
#
#         label = train_df['labels'][i]
#
#         for j in range(len(label)):
#             temp_dict[label[j]] += 1
#
#     for i, label in enumerate(temp_dict.keys()):
#         label_tag_dict[label] = i
#         tag_label_dict[i] = label
#
#     for i, text in enumerate(train_df['norm_text']):
#         label = ast.literal_eval(train_df['labels'][i])
#         line_label = []
#
#         for j in range(len(label)):
#             line_label.append(label_tag_dict[label[j]])
#
#         labels.append(line_label)
#
#     print("text length: " + str(len(texts)))
#     print("label length: " + str(len(labels)))
#     print(temp_dict)
#     print(label_tag_dict)
#     print(tag_label_dict)
#
#     return texts, labels, label_tag_dict, tag_label_dict
#
#
#
# def load_twitter_test(filename):
#     train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')
#
#     texts = []
#     for i, text in enumerate(train_df['norm_text']):
#         text = ast.literal_eval(text)
#         texts.append(text)
#
#     print("text length: " + str(len(texts)))
#
#     return texts