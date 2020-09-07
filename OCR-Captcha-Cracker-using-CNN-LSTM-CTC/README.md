# OCR-Captcha-Cracker-using-CNN-LSTM-CTC

This is a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing CTC loss.

he NN for such use-cases usually consists of convolutional layers (CNN) to extract a sequence of features and recurrent layers (RNN) to propagate information through this sequence. It outputs character-scores for each sequence-element, which simply is represented by a matrix.
This metric is CTC or  Connectionist Temporal Classification Now, there are two things we want to do with this matrix:

* Train: calculate the loss value to train the NN
* Infer: decode the matrix to get the text contained in the input image



We could train a NN to output a character-score for each horizontal position. However, there are two problems with this naive solution:

* it is very time-consuming (and boring) to annotate a data-set on character-level.

* we only get character-scores and therefore need some further processing to get the final text from it. A single character can span multiple horizontal positions, e.g. we could get “ ttooo” because the “o” is a wide character. We have to remove all duplicate “t”s and “o”s. But what if the recognized text would have been “too”? Then removing all duplicate “o”s gets us the wrong result

CTC solves both problems for us:

* we only have to tell the CTC loss function the text that occurs in the image. Therefore we ignore both the position and width of the characters in the image.

* no further processing of the recognized text is needed.


The NN-training will be guided by the CTC loss function. We only feed the output matrix of the NN and the corresponding ground-truth (GT) text to the CTC loss function. Instead, it tries all possible alignments of the GT text in the image and takes the sum of all scores. This way, the score of a GT text is high if the sum over the alignment-scores has a high value.

**Loss calculation**

We need to calculate the loss value for the training samples (pairs of images and GT texts) to train the NN. You already know that the NN outputs a matrix containing a score for each character at each time-step




