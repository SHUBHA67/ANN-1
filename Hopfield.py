import numpy as np

class Hopfield:
    def __init__(self, n_neurons):
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for pattern in patterns:
            self.W += np.outer(pattern, pattern)
            np.fill_diagonal(self.W, 0)
    
    def predict(self, pattern):
        energy = -0.5*((pattern @ self.W)@pattern)
        return np.sign((pattern @ self.W) + energy)        
            





patterns = np.array([[1,1,-1,-1],
                     [-1,-1,1,1],
                     [1,-1,1,-1],
                     [-1,1,-1,1]])

n_neurons = 4

network = Hopfield(n_neurons)
network.train(patterns)

for pattern in patterns:
    prediction = network.predict(pattern)
    print("Input Pattern:", pattern)
    print("Predicted pattern:", prediction)