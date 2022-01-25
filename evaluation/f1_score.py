

#Twitter and Pumbed:

import re
import os
import pickle
import numpy as np
import approximateMatch

def predict_score(pred, toks, y_true, pred_dir, i2l, padlen, metafile=0, fileprefix=''):
    N = len(toks)

    # If the name of a metafile is passed, simply write this round of predictions to file
    if metafile > 0:
        meta = open(metafile, 'a')

    fname = re.sub(r'\\', r'/', os.path.join(pred_dir, fileprefix + 'lstm_softmax_attention_approxmatch_test'))
    with open(fname, 'w') as fout:
        for i in range(N):
            bos = 'BOS\tO\tO\n'
            fout.write(bos)
            if metafile > 0:
                meta.write(bos)

            sentlen = len(toks[i])
            startind = padlen - sentlen

            # preds = [i2l[j] for j in pred[i][startind:]]
            # actuals = [i2l[j] for j in y_true[i][startind:]]

            preds = [i2l[j] for j in pred[i][:sentlen]]
            actuals = [i2l[j] for j in y_true[i][:sentlen]]


            for (w, act, p) in zip(toks[i], actuals, preds):
                line = '\t'.join([w, act, p]) + '\n'
                fout.write(line)
                if metafile > 0:
                    meta.write(line)

            eos = 'EOS\tO\tO\n'
            fout.write(eos)
            if metafile > 0:
                meta.write(eos)
    scores = approximateMatch.get_approx_match(fname)
    # scores['loss'] = test_loss
    if metafile > 0:
        meta.close()

    with open(fname, 'a') as fout:
        fout.write('\nTEST Approximate Matching Results:\n  ADR: Precision ' + str(scores['p']) + ' Recall ' + str(
            scores['r']) + ' F1 ' + str(scores['f1']))
    return scores


predir = './result'
fileprefix = './result'
best_scores_f1=0.0
idx2label = {0: 'O', 1: 'B-ADR', 2: 'I-ADR', 3: 'B-Indication', 4: 'I-Indication'}
pickle_file1 = os.path.join("pickle", "twitter_BIO.pickle3")
train_texts, train_labels, test_texts, test_labels, label_tag_dict, tag_label_dict = pickle.load(open(pickle_file1, 'rb'))

train_ids = []
test_ids = []
train_targets_ids = []
test_targets_ids = []
tag_to_ix = {'O': 0, 'B-ADR': 1, 'I-ADR': 2, 'B-Indication': 3, 'I-Indication': 4}

for i in range(len(test_labels)):
    targets = [tag_to_ix[t] for t in test_labels[i]]
    test_targets_ids.append(targets)

length = [len(seq) for seq in test_targets_ids]
maxlen = np.max(length)


