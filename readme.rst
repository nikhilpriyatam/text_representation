Text Representation
===================

This repository implements the following document embedding methods.

 * TF-IDF
 * Averaged Word Embedding
 * Averaged Sentence Embedding
 * Elmo embedding (to be used for encoding sentences or phrases)
 * Averaged Elmo embedding (can be used to encode documents spanning multiple sentences)
 * Universal sentence encoder
 * Averaged universal sentence encoding
 * uncased BERT model.

 Each of the above methods have two variants. The first variant encodes the sequence of documents in one go. The second variant encodes the documents in a bacth-wise manner.
