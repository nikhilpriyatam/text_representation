# imports
import text_emb as te
import batch_emb as be
import numpy as np
import importlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.metrics import classification_report


if __name__ == '__main__':
    importlib.reload(te)
    importlib.reload(be)

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
    texts, labels = dataset.data, dataset.target
    tr_texts, tr_labels, te_texts, te_labels = texts[:1500], labels[:1500], texts[1500:], labels[1500:]

    # Check Elmo embedding
    x_tr = te.bert_emb(tr_texts, device_map='2')
    x_te = te.bert_emb(te_texts, device_map='2')

    linear_svm = SVC(kernel='linear')
    non_linear_svm = SVC(kernel='rbf')

    non_linear_svm.fit(x_tr, tr_labels)
    pred = non_linear_svm.predict(x_te)
    print(classification_report(pred, te_labels))

    linear_svm.fit(x_tr, tr_labels)
    pred = linear_svm.predict(x_te)
    print(classification_report(pred, te_labels))
