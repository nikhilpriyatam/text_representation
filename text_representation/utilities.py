"""This module contains the utility functions which are called by other modules
of text representation package

@author: Nikhil Pattisapu, iREL, IIIT-H.
"""

import os
import spacy
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf


def batchify(inp, batch_size):
    """A utility function. Batchifies a given python sequence or numpy matrix"""

    if hasattr(inp, "__len__"):
        size = len(inp)
    else: # Assume its a numpy array
        size = inp.shape[0]
    return [inp[i: i + batch_size] for i in range(0, size, batch_size)]


def avg_vectors(matrix, row_ids):
    """Returns the document embedding given the sentence embedding matrix
    consisting of one sentence vecter per row.

    :param matrix: sentence embedding matrix. Each row cosists of sentence
     embedding. The matrix spans the sentence embeddings of several documents.
    :type matrix: A 2D numpy matrix.
    :param row_ids: A list containing the number of sentences in each document.
    :type row_ids: List of integers.
    :return: Document embedding matrix (using averaged sentence embedding)
    :rtype: A 2D numpy matrix.
    """

    res, idx = [], 0
    for row_id in row_ids:
        mat = matrix[idx: idx + row_id]
        res.append(np.mean(mat, axis=0))
        idx = row_id
    return np.vstack(res)


def get_spacy_sent_tokenizer():
    """A Utility function which returns the minimalist spacy model with just
    sentence tokenizer"""

    lang = "en"
    cls = spacy.util.get_lang_class(lang)
    nlp = cls()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    return nlp


def get_sentences(texts, max_sent_len):
    """Tokenizes text and returns sentences as well as number of sentences per
    sample

    :param texts: A list of phrases or texts
    :type texts: A list of strings
    :param max_sent_len: Maximum number of words per sentence. Sentences longer
     than this would be split into multiple sentences.
    :type max_sent_len: int
    :return: A tuple consisting of 2 lists. First one cosists of list of
     tokenized sentences. The second one consists of number of sentences in
     each document or text
    :rtype: Tuple
    """

    nlp = get_spacy_sent_tokenizer()

    texts_sent, texts_nsent = [], []
    for text in texts:
        truncated_sents = []
        sents = [sent.text for sent in nlp(text).sents]

        # If a sentence consists of more than max_sent_len words, split it into
        # multiple sentences.
        for sent in sents:
            words = word_tokenize(sent)
            if len(words) < max_sent_len:
                truncated_sents.append(sent)
            else:
                small_sents = batchify(words, batch_size=max_sent_len)
                for small_sent in small_sents:
                    small_sent = ' '.join(small_sent)
                    truncated_sents.append(small_sent)

        # Each sentence in truncated_sents contains less than max_sent_len
        # words.
        texts_sent += truncated_sents
        texts_nsent.append(len(truncated_sents))
    return texts_sent, texts_nsent


def run_tfhub(texts, tf_model, tf_placeholder, tf_batch_size, gpu_id=None):
    """Run the Tensorflow model and return the text embedding

    :param texts: A list of phrases or texts
    :type texts: A list of strings
    :param tf_model: A tensorflow model comprising of the graph which needs to
     be run.
    :type tf_model: Tensorflow hub model.
    :param tf_placeholder: A placeholder for text
    :type tf_placeholder: tf.string
    :param tf_batch_size: The number of textual items to be processed in a
     single batch
    :type tf_batch_size: int.
    :param gpu_id: The GPU IDs which are to be used for this computation
    :type gpu_id: string
    :return: The text representation.
    :rtype: A numpy matrix.
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        emb = []
        tf_batch_texts = batchify(texts, tf_batch_size)
        for tf_batch in tf_batch_texts:
            var = sess.run(tf_model, {tf_placeholder: tf_batch})
            emb.append(var)
        return np.vstack(emb)
