import os
import sys
import logging

import ast
import pandas as pd
from collections import defaultdict
import pickle
import pickle
import pandas as pd


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
            line_label.append(label_tag_dict[label[j]])

        labels.append(line_label)

    print("text length: " + str(len(texts)))
    print("label length: " + str(len(labels)))
    print(temp_dict)
    print(label_tag_dict)
    print(tag_label_dict)

    return texts, labels, label_tag_dict, tag_label_dict


def load_train_file11(filename):
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
            #line_label.append(label_tag_dict[label[j]])
            line_label.append(label[j])
        labels.append(line_label)

    return texts, labels,label_tag_dict, tag_label_dict

def load_test_file(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)
        texts.append(text)

    print("text length: " + str(len(texts)))

    return texts


asu_train_file = os.path.join('corpus', 'train', 'asu_train')
asu_test_file = os.path.join('corpus', 'test', 'asu_test')

asu_train_texts, asu_train_labels, label_tag_dict, tag_label_dict = load_train_file11(asu_train_file)
asu_test_texts, asu_test_labels, _, _ = load_train_file11(asu_test_file)

print("t9")
print(asu_train_texts)


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

print("80")
print(len(train_texts),len(train_labels))
print(train_texts,train_labels)
print("990")
print(len(test_texts),len(test_labels))
print(test_texts,test_labels)


if __name__ == '__main__':


    def ADR_BIO(labels12):
      for i in range(len(labels12)):
        flagList = [0] * len(labels12[i])
        for j in range(0, len(labels12[i])):
          if labels12[i][j] == "I-ADR" and flagList[j] == 0:
            labels12[i][j] = "B-ADR"
            for k in range(j + 1, len(labels12[i])):
              if labels12[i][k] == "I-ADR":
                flagList[k] = 1
              else:
                break
      for i in range(len(labels12)):
          flagList = [0] * len(labels12[i])
          for j in range(0, len(labels12[i])):
              if labels12[i][j] == "I-Indication" and flagList[j] == 0:
                  labels12[i][j] = "B-Indication"
                  for k in range(j + 1, len(labels12[i])):
                      if labels12[i][k] == "I-Indication":
                          flagList[k] = 1
                      else:
                          break
      return labels12

    labels122=ADR_BIO(train_labels)

    labels122_test = ADR_BIO(test_labels)

    test_la=[]
    print("9990")

    for l in labels122_test:
        for l1 in l:
            test_la.append(l1)
    print("99test")
    print(test_la)

    print(labels122_test)

    def laop(labels122):
        label_tag_dict1 = {}
        tag_label_dict1= {}
        temp_dict1 = defaultdict(float)
        line_label=[]
        labels11=[]
        for label in labels122:
            for j in range(len(label)):
                    temp_dict1[label[j]] += 1

        for i, label in enumerate(temp_dict1.keys()):
            label_tag_dict1[label] = i
            tag_label_dict1[i] = label

        for label in labels122:
            for j in range(len(label)):
                line_label.append(label_tag_dict1[label[j]])
        labels11.append(line_label)

        return labels11,label_tag_dict1,tag_label_dict1

    labels11, label_tag_dict1, tag_label_dict1=laop(labels122)
    print("test55")
    print(labels122)
    labels11_test, label_tag_dict1_test, tag_label_dict1_test = laop(labels122_test)
    print("train_lo")
    print(train_texts,labels11, label_tag_dict1, tag_label_dict1)
    print("hui")
    print(test_texts,labels11_test, label_tag_dict1_test, tag_label_dict1_test)
    pickle_file="./pickle/twitter_BIO.pickle3"
    pickle.dump([train_texts,labels122,test_texts,labels122_test,label_tag_dict1, tag_label_dict1],open(pickle_file, 'wb'))








