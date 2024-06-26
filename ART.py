import numpy as np

class ARTNetwork:
    def __init__(self, input_size, vigilance):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = np.zeros((input_size,1))

    def train(self, input_data,epochs):
        normalized_input = input_data / np.linalg.norm(input_data)
        for _ in range(epochs):
            similarity = normalized_input @ self.weights
            if similarity.any() >= self.vigilance:
                return
            self.weights = np.maximum(self.weights, normalized_input)
            normalized_input = self.weights / np.linalg.norm(self.weights)

    def predict(self, input_data):
        normalized_input = input_data / np.linalg.norm(input_data)
        similarity = normalized_input @ self.weights
        if similarity.any() >= self.vigilance:
            output_pattern = np.zeros(len(similarity))
            print(similarity)
            winner = np.argmax(similarity)
            output_pattern[winner] = 1
        else:
            output_pattern = np.zeros(len(similarity) + 1)
            output_pattern[-1] = 1
        return output_pattern

# Usage example
input_size = 4
vigilance = 0.9
art = ARTNetwork(input_size, vigilance)

# Training data
data = np.array([[1, 1, 0, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 1],
                 [1, 0, 1, 0]])

# Train the ART network
for sample in data:
    art.train(sample,1000)

# Test a new input
new_input = np.array([0, -1, 0, 0])
new_output = art.predict(new_input)
print(new_output)