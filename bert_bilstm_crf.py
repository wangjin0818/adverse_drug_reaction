import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from transformers import BertTokenizerFast, TFBertModel

from crf import CRFModel

epochs = 1
batch_size = 4
learning_rate = 2e-5
decay_factor = 1e-6
hidden_state = 128

pickle_file = os.path.join('data', 'data_pickle', 'twitter_BIO.pickle3')
train_texts, train_tags, test_texts, test_tags, tag2id, id2tag = pickle.load(open(pickle_file, 'rb'))
unique_tags = set(tag for doc in train_tags for tag in doc)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding="max_length",
                            truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding="max_length",
                           truncation=True)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


train_labels = encode_tags(train_tags, train_encodings)
test_labels = encode_tags(test_tags, test_encodings)

train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
test_encodings.pop("offset_mapping")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels,
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

print(len(train_encodings[0]))
print(len(train_labels[0]))
print(train_labels[0])

bert_model = TFBertModel.from_pretrained("bert-base-uncased")
bert_config = bert_model.config

input_ids = tf.keras.Input(shape=(bert_config.max_position_embeddings,), dtype=tf.int32, name='input_ids')
token_type_ids = tf.keras.Input(shape=(bert_config.max_position_embeddings,), dtype=tf.int32, name='token_type_ids')
attention_mask = tf.keras.Input(shape=(bert_config.max_position_embeddings,), dtype=tf.int32, name='attention_mask')
inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

# bert_model
last_hidden_state = bert_model(inputs).last_hidden_state

# LSTM
lstm_hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_state, return_sequences=True))(
    last_hidden_state)
print(lstm_hidden.shape)

base = tf.keras.Model(inputs=inputs, outputs=lstm_hidden)

# CRF
crf_model = CRFModel(base, len(unique_tags) + 1)  # plus -100

# model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimier = AdamW(learning_rate=learning_rate, weight_decay=decay_factor)
crf_model.compile(optimizer=optimier)

print(crf_model.summary())
crf_model.fit(train_dataset.shuffle(1000).batch(batch_size), validation_data=test_dataset.batch(batch_size),
              epochs=epochs,
              batch_size=batch_size)

y_test_pred = crf_model.predict(test_dataset.batch(batch_size))
print(y_test_pred)