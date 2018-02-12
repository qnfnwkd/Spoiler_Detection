# A Deep Nueral Spoiler Detection Model using a Genre-Aware Attention Mechanism (PAKDD'18)
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
