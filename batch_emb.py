"""Return vectorized batch-wise representation of a given textual docs or
phrases. Uses scikit-learn, spacy and tensorflow-hub modules. Most useful
tensorflow hub URLs are mentioned below.

https://tfhub.dev/google/elmo/2
https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
https://tfhub.dev/google/universal-sentence-encoder-large/3

@author: Nikhil Pattisapu, iREL, IIIT-H.
"""


import os
import subprocess
import signal
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_serving.client import BertClient
import utilities as utils
import spacy
import tensorflow as tf
import tensorflow_hub as hub


def avg_word_emb(texts, nlp=None, model_type="en_core_web_lg",
                 n_vectors=None, batch_size=32):
    """Yields averaged word embedding based representation of batches of texts.

    :param texts: list of documents.
    :type texts: list of strings.
    :param nlp: Spacy NLP Model preloaded in memory
    :type nlp: optional, spacy model object. Default is en_core_web_lg
    :param model_type: Model type used for various word embeddings
    :type model_type: string, optional (default word2vec vectors)
    :param n_vectors: Number of vectors to keep
    :type n_vectors: None, int, optional
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy sparse matrix.
    """

    if nlp is None:
        nlp = spacy.load(model_type)
        nlp.disable_pipes('tagger', 'parser', 'ner')
    if n_vectors is not None:
        nlp.vocab.prune_vectors(n_vectors)

    batched_texts = utils.batchify(texts, batch_size)
    for batch_texts in batched_texts:
        embeds = [nlp(text).vector for text in batch_texts]
        yield np.vstack(embeds)


def avg_sent_emb(texts, nlp=None, model_type="en_core_web_lg",
                 n_vectors=None, batch_size=32):
    """Yields averaged word embedding based representation of batches of texts.

    :param texts: list of documents.
    :type texts: list of strings.
    :param nlp: Spacy NLP Model preloaded in memory
    :type nlp: optional, spacy model object. Default is en_core_web_lg
    :param model_type: Model type used for various word embeddings
    :type model_type: string, optional (default word2vec vectors)
    :param n_vectors: Number of vectors to keep
    :type n_vectors: int, optional, default is None
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy sparse matrix.
    """

    if nlp is None:
        nlp = spacy.load(model_type)
        nlp.disable_pipes('tagger', 'parser', 'ner')
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
    if n_vectors is not None:
        nlp.vocab.prune_vectors(n_vectors)

    texts = utils.batchify(texts, batch_size)
    for batch_texts in texts:
        embeds = []
        for text in batch_texts:
            embed = [sent.vector for sent in nlp(text).sents]
            embeds.append(np.mean(np.vstack(embed), axis=0))
        yield np.vstack(embeds)


def elmo_emb(texts, module_url="https://tfhub.dev/google/elmo/2",
             tf_batch_size=64, gpu_id=None, batch_size=32):
    """Yields Elmo representation of batches of phrases or texts

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param module_url: TFHub module URL for Elmo model.
    :type module_url: String, optional. Default is
     https://tfhub.dev/google/elmo/2
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Elmo Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy sparse matrix.
    """

    elmo = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = elmo(tf_placeholder, signature="default",
                    as_dict=True)["default"]
    batched_texts = utils.batchify(texts, batch_size)
    for batch_texts in batched_texts:
        yield utils.run_tfhub(batch_texts, tf_model, tf_placeholder,
                              tf_batch_size, gpu_id)


def use_emb(texts, module_url=("https://tfhub.dev/google/"
                               "universal-sentence-encoder-large/3"),
            tf_batch_size=64, gpu_id=None, batch_size=32):
    """Yields Universal Sentence Encoder representation of batches of phrases
    or text

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param module_url: TFHub module URL for pre-trained Universal Sentence
     Encoder model.
    :type module_url: String, optional. Default is
     https://tfhub.dev/google/universal-sentence-encoder-large/3
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Universal Sentence Encoder Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy sparse matrix.
    """

    use = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = use(tf_placeholder)

    batched_texts = utils.batchify(texts, batch_size)
    for batch_texts in batched_texts:
        yield utils.run_tfhub(batch_texts, tf_model,
                              tf_placeholder, tf_batch_size, gpu_id)


def bert_emb(texts, model_dir='~/uncased_L-12_H-768_A-12/', batch_size=32,
             **kwargs):
    """Yields BERT representation of batches of phrases or texts.

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param model_dir: The path of pre-trained BERT model.
    :type model_dir: string, optional, default is ~/uncased_L-12_H-768_A-12
    :param kwargs: Keyword arguments to be passed to the server which affect
     the performance of the BERT model (Includes GPU ID, etc). For more options
     visit: https://bert-as-service.readthedocs.io/en/latest/source/server.html
     and https://github.com/hanxiao/bert-as-service
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    # Start server with appropriate parameters
    cmd = 'bert-serving-start -model_dir ' + model_dir + ' '
    for key, val in kwargs.items():
        cmd += '-' + key + ' ' + val + ' '

    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                              preexec_fn=os.setsid)
    bert_client = BertClient()
    batched_texts = utils.batchify(texts, batch_size)
    for batch_texts in batched_texts:
        embeddings = bert_client.encode(batch_texts)
        yield embeddings

    os.killpg(os.getpgid(server.pid), signal.SIGTERM)


def avg_elmo_emb(texts, module_url="https://tfhub.dev/google/elmo/2",
                 tf_batch_size=64, gpu_id=None, batch_size=32):
    """Yields averaged sentence Elmo representation of batches of phrases
    or text

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param module_url: TFHub module URL for Elmo model.
    :type module_url: String, optional. Default is
     https://tfhub.dev/google/elmo/2
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Elmo Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    elmo = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = elmo(tf_placeholder, signature="default",
                    as_dict=True)["default"]
    batched_texts = utils.batchify(texts, batch_size)
    nlp = spacy.load("en")
    for batch_texts in batched_texts:
        texts_sent, texts_nsent = utils.get_sentences(batch_texts, nlp)
        embeddings = utils.run_tfhub(texts_sent, tf_model, tf_placeholder,
                                     tf_batch_size, gpu_id)
        yield utils.avg_vectors(embeddings, texts_nsent)


def avg_use_emb(texts, module_url="https://tfhub.dev/google/\
                universal-sentence-encoder-large/3", tf_batch_size=64,
                gpu_id=None, batch_size=32):
    """Yields averaged sentence universal sentence encoding representation of
    batches of phrases or texts.

    :param texts: list of documents or phrases
    :type texts: list of strings
    :param module_url: TFHub module URL for pre-trained Universal Sentence
     Encoder model.
    :type module_url: String, optional. Default is
     https://tfhub.dev/google/universal-sentence-encoder-large/3
    :param tf_batch_size: Number of strings to be processed in a single
     batch of Universal Sentence Encoder Tensorflow Hub model.
    :type tf_batch_size: int, optional, default is 64
    :param gpu_id: The default GPU ID which has to be used to run Elmo
     pre-trained model
    :type gpu_id: string (to be used by os.environ['CUDA_VISIBLE_DEVICES'].
     optional, Default is None (Use System specified value)
    :param batch_size: The batch size of the representation
    :type batch_size: int, optional, default is 32
    :return: A matrix representing the texts
    :rtype: A numpy matrix.
    """

    use = hub.Module(module_url, trainable=False)
    tf_placeholder = tf.placeholder(tf.string)
    tf_model = use(tf_placeholder)

    batched_texts = utils.batchify(texts, batch_size)
    nlp = spacy.load("en")
    for batch_texts in batched_texts:
        texts_sent, texts_nsent = utils.get_sentences(batch_texts, nlp)
        embeddings = utils.run_tfhub(texts_sent, tf_model, tf_placeholder,
                                     tf_batch_size, gpu_id)
        yield utils.avg_vectors(embeddings, texts_nsent)
