# Note: "Deep Reinforcement Learning: An Overview"


## Introduction

This is a note of [Deep Reinforcement Learning: An Overview, Yuxi Li](https://arxiv.org/pdf/1701.07274.pdf).

This note only lists the most important concepts in this paper. Most of the idea may not be eliborated. Only key words are mentioned in the note.

To understand more about the details of reinforcement learning, please scrutinize the paper.

## Note

### Machine Learning

#### Category

* Supervised learning

  * Classification 
  * Regression

* Unsupervised learning: representation learning 

  * Clustering 
  * Density estimation

* Reinforcement learning

#### Elements

* Dataset

  * Non-overlapping training, validation and testing subsets

* Cost/Loss function

  * Category

    * Training error measures the error on the training data
    * Generalization error, or test error, measures the error on new input data
	
  * Measurement
  
    The following measurements are equivalent.
  
    * Maximum likelihood estimation (MLE)
	* Minimize KL divergence
	* Minimize the cross-entropy
	* negative log-likelihodd (NLL)

* Optimization procedure

  * Gradient descent 
  * Stochastic gradient descent

* Model











