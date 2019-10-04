# import keras
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split

# cancer = load_breast_cancer()
# x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target, test_size=0.2)

# model = keras.Sequential()
# model.add(keras.layers.Dense(32))
# model.add(keras.layers.Dense(10))
# model.add(keras.layers.Dense(1))
# model.compile(optimizer='sgd',loss='binary_crossentropy')
# model.fit(x_train,y_train)
# model.summary()
# model.evaluate(x_test,y_test)
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
sentences = ['I like my cats',\
            'dogs at the best',\
            'my cat is always knocking things over',\
            'dogs are mans best friend']
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(sentences)
x= bow.toarray()

y = np.array([[1],[0],[1],[0]])

class LogsticRegression(nn.Module):
    def __init__(self,dims):
        super(LogisticRegression).__init__()
        self.L1 = nn.Linear(dims)
        self.sigmoid = nn.Sigoid()

    def forward(self,x):
        z1 = self.L1(x)
        a1 = self.Sigmoid(z1)
        return a1


model = LogsticRegression(x.shape[1])
optimizer = optim.SGD(model.parameters(),lr=0.1)
loss = nn.BCELoss()
for epoch in range(100):
    model.zero_grad()
    model.forward(x)
    output = loss(yhat,y)
    output.backward()
    optimizer.step()


