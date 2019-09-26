**AIT726**
******************************
**HW1** <br>
In this assignment, you will build a naïve Bayes and a logistic regression classifier for sentiment
classification. We are defining sentiment classification as two classes: positive and negative.
Our data set consists of movie reviews. The zip directory for the data contains training and test
datasets, where each file contains one movie review. You will build the model using training
data and evaluate with test data. Training data contains 25000 reviews and test data contains
25000 reviews. <br>
Command to run the file: python HW1.py <br>
i. main - runs all of the functions
    ii. get_trainandtest_vocabanddocs() - converts dataset into tokens (stemmed and unstemmed), creates megatraining document and extracts vocabulary
    iii. get_vectors() - creates BOW and TFIDF vectors for test and train both stemmed and unstemmed
    iv. get_class_priors() - calculates the class prior likelihoods for use in Naive Bayes predictions
    v. get_perword_likelihood() - calculates dictionaries for each feature vector to be used in the Naive Bayes prediction calculation
    vi. predict_NB() - predicts the class of all of the test documents for all of the feature vectors using Naive Bayes
    vii. evaluate - returns accuracy and confusion matrix for predictions 
    viii. Logistic_Regression_L2_SGD - logistic regression model class used to create the model and form predictions on test vectors
<br>
Due to the size of the dataset, and the number of tokens we are required to keep, many of the operations when creating vectors utilize a large amount of RAM. 
This code was tested on a machine with 64GB of DDR4 RAM. Variables are deleted throughout when they are not needed to save memory. Needed data structures are saved and loaded
for later use.