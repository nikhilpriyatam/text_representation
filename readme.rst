Text Representation
===================

Installation
------------

`pip install -r requirements.txt`
`pip install text_representation-0.1-py3-none-any.whl`


Methods
-------

This repository implements the following document embedding methods.

 * TF-IDF
 * Averaged Word Embedding
 * Averaged Sentence Embedding
 * Elmo embedding (to be used for encoding sentences or phrases)
 * Averaged Elmo embedding (can be used to encode documents spanning multiple sentences)
 * Universal sentence encoder
 * Averaged universal sentence encoding
 * uncased BERT model based encodings.

Each of the above methods have two variants. The first variant encodes the sequence of documents in one go. The second variant encodes the documents in a batch-wise manner.


More Help
---------

  * This code depends on a server which serves BERT embeddings.
  * More about that can be found `in this repo<>`_.
  * It can also be `installed using pip<https://pypi.org/project/bert-serving-server/>`_.
  * The complete documentation is also available `Get started here <https://bert-as-service.readthedocs.io/en/latest/section/get-start.html>`_
  * You can create a pip-installable package from this repository by issuing the following command in the main directory of the repo.
    
    * `pip setup.py bdist_wheel`.
    * The resultant wheel file will be in `dist` folder.
    * For system-wide install use `pip install wheel_file_name`
    * Delete any additional folders or files generated
