import numpy as np

input_data = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1],  # 1
    [0, 0, 0, 0, 0, 0, 1, 0],  # 2
    [0, 0, 0, 0, 0, 0, 1, 1],  # 3
    [0, 0, 0, 0, 0, 1, 0, 0],  # 4
    [0, 0, 0, 0, 0, 1, 0, 1],  # 5
    [0, 0, 0, 0, 0, 1, 1, 0],  # 6
    [0, 0, 0, 0, 0, 1, 1, 1],  # 7
    [0, 0, 0, 0, 1, 0, 0, 0],  # 8
    [0, 0, 0, 0, 1, 0, 0, 1],  # 9
    [0, 0, 0, 0, 1, 0, 1, 0]   # 10
])
targets = [1,0,1,0,1,0,1,0,1,0]

class Perceptron:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def activation(self, Z):
       
        return 1/(1+np.exp(-Z))
    
    def predict(self, X):
        pred=[]
        for i in range(len(X)):
            Z = np.dot(X[i], self.W)+self.b
            prob = self.activation(Z)

            if prob>=0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred
    
    def train(self,X,y,epochs=100):
        self.W = np.random.rand(len(X[0]))
        self.b = np.random.rand()
        for epoch in range(epochs):
            for i in range(len(X)):
                Z = np.dot(X[i],self.W) + self.b
                preds = self.activation(Z)

                error = y[i]-preds

                self.W += self.lr * error * X[i]
                self.b += self.lr * error


P = Perceptron()
P.train(input_data, targets, epochs=100)

pred = P.predict(input_data)
print(pred == targets)

binary_numbers_11_to_20 = [
    [0, 0, 0, 0, 1, 0, 1, 1],  # 11
    [0, 0, 0, 0, 1, 1, 0, 0],  # 12
    [0, 0, 0, 0, 1, 1, 0, 1],  # 13
    [0, 0, 0, 0, 1, 1, 1, 0],  # 14
    [0, 0, 0, 0, 1, 1, 1, 1],  # 15
    [0, 0, 0, 1, 0, 0, 0, 0],  # 16
    [0, 0, 0, 1, 0, 0, 0, 1],  # 17
    [0, 0, 0, 1, 0, 0, 1, 0],  # 18
    [0, 0, 0, 1, 0, 0, 1, 1],  # 19
    [0, 0, 0, 1, 0, 1, 0, 0]   # 20
]     
values = P.predict(binary_numbers_11_to_20)           
print(values)
