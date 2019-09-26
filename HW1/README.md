AIT726
=======
HW1
-------
In this assignment, you will build a naïve Bayes and a logistic regression classifier for sentiment
classification. We are defining sentiment classification as two classes: positive and negative.
Our data set consists of movie reviews. The zip directory for the data contains training and test
datasets, where each file contains one movie review. You will build the model using training
data and evaluate with test data. Training data contains 25000 reviews and test data contains
25000 reviews. <br>
**Models:**<br>
    * Naive Bayes<br>
    * Logistic Regression<br>
**Vectors:** <br>
    * Bag of Words - Frequency<br>
    * Bag of Words - Binary (is word present in document)<br>
    * Term Frequency Inverse Document Frequency<br>
**Preprocessing:**<br>
    * Stemmed<br>
    * Unstemmed<br>
**Command to run the file:** <br>
`python HW1.py `<br>

>i. main - runs all of the functions <br>
    ii. get_trainandtest_vocabanddocs() - converts dataset into tokens (stemmed and unstemmed), creates megatraining document and extracts vocabulary<br>
    iii. get_vectors() - creates BOW and TFIDF vectors for test and train both stemmed and unstemmed<br>
    iv. get_class_priors() - calculates the class prior likelihoods for use in Naive Bayes predictions<br>
    v. get_perword_likelihood() - calculates dictionaries for each feature vector to be used in the Naive Bayes prediction calculation<br>
    vi. predict_NB() - predicts the class of all of the test documents for all of the feature vectors using Naive Bayes<br>
    vii. evaluate - returns accuracy and confusion matrix for predictions <br>
    viii. Logistic_Regression_L2_SGD - logistic regression model class used to create the model and form predictions on test vectors<br>

<br>
Due to the size of the dataset, and the number of tokens we are required to keep, many of the operations when creating vectors utilize a large amount of RAM. <br>
This code was tested on a machine with 64GB of DDR4 RAM. Variables are deleted throughout when they are not needed to save memory. Needed data structures are saved and loaded for later use. <br>

Results:
--------

Model | Accuracy
:------------: | :-------------:
NB-NOSTEM-FREQ| 80.8
NB-NOSTEM-BINARY | 81.6
NB-NOSTEM-TFIDF | 71.26
NB-STEM-FREQ | 80.272
NB-STEM-BINARY | 80.86
NB-STEM-TFIDF | 68.968
LOGISTIC_FREQ_NOL2 | 82.488
LOGISTIC_TFIDF_NOL2 | 88.588
LOGISTIC_FREQ_STEM_NOL2 | 61.508
LOGISTIC_BINARY_STEM_NOL2 | 83.18
LOGISTIC_TFIDF_STEM_NOL2 | 88.144
LOGISTIC_FREQ_L2 | 55.316
LOGISTIC_BINARY_L2 | 78.18
LOGISTIC_TFIDF_L2 | 86.068
LOGISTIC_FREQ_STEM_L2 | 58.36
LOGISTIC_BINARY_STEM_L2 | 79.732
LOGISTIC_TFIDF_STEM_L2 | 85.592

*****************************
