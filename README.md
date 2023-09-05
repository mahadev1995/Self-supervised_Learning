# Visual Word Embedding: A Self-Supervised Learning Approach for Image Embeddings

Self-supervised learning approaches can lessen the need for large-scale labeled datasets when using deep learning techniques to solve computer visionrelated tasks. In this work, we propose a self-supervised learning method - “Visual Word Embeddings”. 

We divide our work into two parts. In the first part, we develop an asymmetric encoder and decoder architecture. Our approach uses a towered architecture for the encoder and a straightforward convolutional model for the decoder, which reconstructs the target from the latent representation of the context received from the encoder.

##### Schematic diagram of the model
![model](https://user-images.githubusercontent.com/51476618/220179502-0a945752-8cd8-48ba-8235-853f4a6d8c44.png)

In the second part, we assess the performance of the model using the linear evaluation approach. For this, we use the learned representation to perform a downstream
image classification task. Finally, we compare the performance of our model with that of a baseline simple autoencoder.
##### Schematic diagram of the linear evaluation process
![LinearEvaluation](https://user-images.githubusercontent.com/51476618/220179854-08682c4e-cc6e-4cdc-98d3-21642d630f95.png)
