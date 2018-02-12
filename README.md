# A Deep Nueral Spoiler Detection Model using a Genre-Aware Attention Mechanism (PAKDD'18)
In this repository, I'm going to implement the paper, which is <b>"A Deep Neural Spoiler Detection Model using a Genre-Aware Attention Mechanism"</b>, (B. Chang et al, PAKDD'18), using a Tensorflow library.

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
<img src="/figures/model_description.png" width="600px" height="auto">
</p>
Our spoiler detection model consists of genre encoder and sentence encoder.

## Data set
Data set is available at [here](https://s3.amazonaws.com/poiprediction/instagram.tar.gz). The data set includes "train.txt", "validation.txt", "test.txt", and "visual_feature.npz". The "train.txt"  "validation.txt" "test.txt" files include the training, validation, and tesing data respectively. The data is represented in the following format:
```bash
<post_id>\t<user_id>\t<word_1 word_2 ... >\t<poi_id>\t<month>\t<weekday>\t<hour>
```

All post_id, user_id, word_id, and poi_id are anonymized. Photo information also cannot be distributed due to personal privacy problems. So we relase the converted visual features from the output of the FC-7 layer of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) used as the visual feature extractor. If you want to use other visual feature extractor, such as [GoogleNet](http://arxiv.org/abs/1602.07261), [ResNet](https://arxiv.org/abs/1512.03385), you could implement it on your source code. We use a pre-trained VGGNet16 by [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) The "visual_feature.npz" file contains the visual features where the i-th row denotes i-th post's features.

### statistics
<table style="align=center;">
<tr><td>number of total post</td><td>number of POIs</td><td>number of users</td><td>size of vocabulary</td></tr>
<tr><td>736,445</td><td>9,745</td><td>14,830</td><td>470,374</td></tr>
<tr><td>size of training set</td><td>size of validation set</td><td>size of test set</td></tr>
<tr><td>526,783</td><td>67,834</td><td>141,828</td></tr>
</table>

## Getting Started
The code that implements our proposed model is implemented for the above dataset, which includes pre-processd visual feature. If you want to use a raw image that is not pre-processed, implement VGGNet on your source code as visual CNN layer.

### Prerequisites
- python 2.7
- tensorflow r1.2.1

### Usage
```bash
git clone https://github.com/qnfnwkd/DeepPIM
cd DeepPIM
python train.py
```
