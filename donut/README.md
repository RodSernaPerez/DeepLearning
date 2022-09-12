# DONUT

Info based on the [original paper](https://arxiv.org/pdf/2111.15664.pdf)

This is the model to process documents without needing an OCR.

## Architecture

### Encoder

It is a visual transformer encoder. In the paper they propose Swin Transsformer.

It receives an image, which is split into $n$ non overlapping images. Then the encoder produces as output a matrix $z$ of size $n x d$.


### Decoder

Receives $z$ and returns a sequence $y$ of size $m x v$. The value $v$ is the size of the vocabulary (it will be interpreted as a one hot vector).
$m$ is a hyperparameter.

In the paper they propose [BART](https://arxiv.org/pdf/1910.13461.pdf) for this architecture. 
BART has the same architecture than BERT but only changes in pretraining, so any BERT decoder could be used.
They use a pretrained model for weights initiation.


## Pretraining 

As a pretraining task, they train an OCR: given an image of text, predict the text on it.

The image goes to the encoder and the next word is predicted given the previous in the decoder.

For the data they used:

- IIT-CDIP: a public available dataset
- A technique to generate synthetic data