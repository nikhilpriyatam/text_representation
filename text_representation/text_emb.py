"""Return vectorized representation of a given textual docs or phrases. Uses
scikit-learn, spacy and tensorflow-hub modules. Most useful tensorflow hub URLs
are mentioned below.

https://tfhub.dev/google/elmo/2
https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
https://tfhub.dev/google/universal-sentence-encoder-large/3

@author: Nikhil Pattisapu, iREL, IIIT-H.
"""


import os
import signal
import subprocess
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import text_representation.utilities as utils
import spacy
from bert_serving.client import BertClient
import tensorflow as tf
import tensorflow_hub as hub


def tfidf(texts, vectorizer=None):
    """Return the TF-IDF representation corresponding to texts.

    :param texts: list of documents.
    :type texts: list of strings.
    :param vectorizer: A fitted Sklearn's TFIDF vectorizer
    :type vectorizer: optional,
     sklearn.feature_extraction.text.TfidfVectorizer
    :return: A matrix representing the texts
    :rtype: A numpy sparse matrix.
    """

    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts)

    res = vectorizer.transform(texts)
    return res, vectorizer


def avg_word_emb(texts, nlp=None, model_type="en_core_web_lg", n_vectors=None):
    """Return averaged word embedding based representation of texts.

    :param texts: list of documents.
    :type texts: list of strings.
    :param nlp: Spacy NLP Model preloaded in memory
    :type nlp: optional, spacy model object.
    :param model_type: Model type used for various word embeddings
    :type model_type: string, optional (default word2vec vectors)
    :param n_vectors: Number of vectors to keep
    :type n_vectors: None, int, optional
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    if nlp is None:
        nlp = spacy.load(model_type)
        nlp.disable_pipes('tagger', 'parser', 'ner')
    if n_vectors is not None:
        nlp.vocab.prune_vectors(n_vectors)

    embeds = [nlp(text).vector for text in texts]
    return np.vstack(embeds)


def avg_sent_emb(texts, nlp=None, model_type="en_core_web_lg", n_vectors=None):
    """Return averaged sentence embedding based representation of texts. A
    sentence embedding is obtained by averaging the word embeddings of the
    words present in it.

    :param texts: list of documents.
    :type texts: list of strings.
    :param nlp: Spacy NLP Model preloaded in memory
    :type nlp: optional, spacy model object.
    :param model_type: Model type used for various word embeddings
    :type model_type: string, optional (default word2vec vectors)
    :param n_vectors: Number of vectors to keep. Setting a small value results
     in faster speed at the cost of reduced accuracy. Can be used for quick
     testing.
    :type n_vectors: None, int, optional
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    if nlp is None:
        nlp = spacy.load(model_type)
        nlp.disable_pipes('tagger', 'parser', 'ner')
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
    if n_vectors is not None:
        nlp.vocab.prune_vectors(n_vectors)

    embeds = []
    for text in texts:
        embed = [sent.vector for sent in nlp(text).sents]
        embeds.append(np.mean(np.vstack(embed), axis=0))
    return np.vstack(embeds)


def elmo_emb(texts, module_url="https://tfhub.dev/google/elmo/2",
             tf_batch_size=64, gpu_id=None):
    """Return Elmo representation of the phrase or text

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Elmo Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    elmo = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = elmo(tf_placeholder, signature="default",
                    as_dict=True)["default"]
    return utils.run_tfhub(texts, tf_model, tf_placeholder, tf_batch_size,
                           gpu_id)


def use_emb(texts, module_url=("https://tfhub.dev/google/"
                               "universal-sentence-encoder-large/3"),
            tf_batch_size=64, gpu_id=None):
    """Return Universal Sentence Encoder representation of the phrase or text

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Universal Sentence Encoder Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :return: A matrix representing the texts.
    :rtype: A numpy matrix.
    """

    use = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = use(tf_placeholder)
    return utils.run_tfhub(texts, tf_model, tf_placeholder, tf_batch_size,
                           gpu_id)


def bert_emb(texts):
    """Return BERT representation of the phrase or text. In order for this
    function to return embeddings it needs to meet the following prerequisite.
    There should be a server running in the background which serves bert
    embedding. The commands for starting and terminating the servers are
    mentioned below

        * bert-serving-start -model_dir=~/uncased_L-12_H-768_A-12
        * bert-serving-terminate -port_in 5555.

    For other useful options look at bert-serving-start --help and
    bert-serving-terminate --help.

    In order to install the above mentioned server please visit to the
    following sources:

        * https://bert-as-service.readthedocs.io/en/latest/source/server.html
        * https://github.com/hanxiao/bert-as-service

    :param texts: list of documents or phrases
    :type texts: list of strings
    :return: A matrix representing the texts.
    :rtype: A numpy matrix.
    """

    # Start server with appropriate parameters
    bert_client = BertClient()
    embeddings = bert_client.encode(texts)
    return embeddings


def avg_elmo_emb(texts, module_url="https://tfhub.dev/google/elmo/2",
                 tf_batch_size=64, gpu_id=None, max_sent_len=40):
    """Return the averaged sentence Elmo representation of the phrase or text.

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Elmo Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param max_sent_len: Maximum number of words per sentence. If a sentence is
     found to be larger than this then it is split into multiple sentences.
    :type max_sent_len: int, optional, default is 40.
    :return: A matrix representing the texts.
    :rtype: A numpy matrix.
    """

    elmo = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = elmo(tf_placeholder, signature="default",
                    as_dict=True)["default"]
    texts_sent, texts_nsent = utils.get_sentences(texts, max_sent_len)
    embeddings = utils.run_tfhub(texts_sent, tf_model, tf_placeholder,
                                 tf_batch_size, gpu_id)
    return utils.avg_vectors(embeddings, texts_nsent)


def avg_use_emb(texts, module_url=("https://tfhub.dev/google/"
                                   "universal-sentence-encoder-large/3"),
                tf_batch_size=64, gpu_id=None, max_sent_len=40):
    """Return averaged sentence universal sentence encoding representation of
    the phrase or text.

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Universal Sentence Encoder Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param max_sent_len: Maximum number of words per sentence. If a sentence is
     found to be larger than this then it is split into multiple sentences.
    :type max_sent_len: int, optional, default is 40.
    :return: A matrix representing the texts.
    :rtype: A numpy matrix.
    """

    use = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = use(tf_placeholder)

    texts_sent, texts_nsent = utils.get_sentences(texts, max_sent_len)
    embeddings = utils.run_tfhub(texts_sent, tf_model, tf_placeholder,
                                 tf_batch_size, gpu_id)
    return utils.avg_vectors(embeddings, texts_nsent)
