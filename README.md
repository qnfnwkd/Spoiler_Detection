# A Deep Nueral Spoiler Detection Model using a Genre-Aware Attention Mechanism (PAKDD'18)
In this repository, We're going to implement the paper, which is <b>"A Deep Neural Spoiler Detection Model using a Genre-Aware Attention Mechanism"</b>, (B. Chang et al, PAKDD'18), using a Tensorflow library.

## Abstract
The fast-growing volume of online activity and user-generated content increases the chances of users being exposed to spoilers. 
To address this problem, several spoiler detection models have been proposed. 
However, most of the previous models rely on hand-crafted domain-specific features, which limits the generalizability of the models. 
In this paper, we propose a new deep neural spoiler detection model that uses a genre-aware attention mechanism. 
Our model consists of a genre encoder and a sentence encoder. 
The genre encoder is used to extract a genre feature vector from given genres using a convolutional neural network. 
The sentence encoder is used to extract sentence feature vectors from a given sentence using a bi-directional gated recurrent unit. 
We also propose a genre-aware attention layer based on the attention mechanism that utilizes genre information for detecting spoilers which vary by genres.
Using a sentence feature, our proposed model determines whether a given sentence is a spoiler.
The experimental results on a spoiler dataset show that our proposed model which does not use hand-crafted features outperforms the state-of-the-art spoiler detection baseline models. 
We also conduct a qualitative analysis on the relations between spoilers and genres, and highlight the results through an attention weight visualization.

## Model description
<p align="center">
<img src="/figures/model_description.png" width="700px" height="auto">
</p>
Our spoiler detection model consists of genre encoder and sentence encoder.

## Getting Started
### Dataset Preparation
We conduct the experimental evaluation on the [dataset](http://umiacs.umd.edu/jbg/downloads/spoilers.tar.gz) used in the paper <b>"Spoiler alert: Machine learning approaches to detect social media posts with revelatory information."</b>, (Boyd-Graber et al, 2013).<br>
We use the WordPunctTokenizer of the nltk to tokenize the sentence, and use the [Glove (840B, 300d)](https://nlp.stanford.edu/projects/glove/) as a pre-trained word embedding.<br>
Since the published dataset does not contain the genre informatoin, we crawl the genre information of the creative works from [IMDb.com](http://www.imdb.com/).<br>
The data is represented in the following format:
```bash
<label> \t <word_1 word_2 word_3 ...> \t <genre_1  genre_2 ...>
```

### Prerequisites
- python 2.7
- tensorflow r1.2.1

