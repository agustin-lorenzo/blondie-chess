import numpy as np
import random
import copy
from math import exp

tau = 0.0839

""" 
def tanh(X):
    return [(exp(x)-exp(-x))/(exp(x)+exp(-x)) for x in X] """

piece2Value = {
    'p': -1,
    'n': -3,
    'b': -3,
    'r': -5,
    'q': -9,
    'k': 0,
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 0
}


def getSubsquares(board, startSize=3):
    rows = 8
    cols = 8
    boardString = str(board).replace("\n", '')
    boardString = boardString.replace(' ', '')
    allSubsquares = []  

    for height in range(startSize, rows + 1):  
        for width in range(startSize, cols + 1): 
            subsquareList = []  

            for row in range(rows - height + 1):
                for col in range(cols - width + 1):
                    subsquare = []

                    for h in range(height):  
                        for w in range(width): 
                            index = (row + h) * cols + (col + w)
                            char = boardString[index]
                            if char in piece2Value:
                                value = piece2Value[char]
                                subsquare.append(value)
                            elif char == '.':
                                value = 0
                                subsquare.append(value)

                    subsquareList.append(subsquare)
            allSubsquares.extend(subsquareList)


    def flattenOutput(subsquares):
        return [value for sublist in subsquares for value in sublist]
    
    return flattenOutput(allSubsquares)


class Network():
    def __init__(self, inputSize = 9604):
        print("initializing network...")
        self.sigma = 0.05
        self.fitness = 0

        self.weightsInputHidden1 = np.random.randn(inputSize, 91) * 0.2 # normalizing values to range [-0.2, 0.2]
        self.biasHidden1 = np.random.randn(1, 91) * 0.2

        self.weightsHidden1Hidden2 = np.random.randn(91, 40) * 0.2
        self.biasHidden2 = np.random.randn(1, 40) * 0.2

        self.weightsHidden2Hidden3 = np.random.randn(40, 10) * 0.2
        self.biasHidden3 = np.random.randn(1, 10) * 0.2

        self.weightsHidden3Output = np.random.randn(10, 1) * 0.2
        self.biasOutput = np.random.randn(1, 1) * 0.2
    
    def forward(self, X, board=None):
        # try:
        #     self.hidden1Input = np.dot(X, self.weightsInputHidden1) + self.biasHidden1
        # except:
        #     print("board: \n", board)
        #     print("len(subsquares): ", len(X))

        self.hidden1Input = np.dot(X, self.weightsInputHidden1) + self.biasHidden1
        self.hidden1Output = np.tanh(self.hidden1Input)

        self.hidden2Input = np.dot(self.hidden1Output, self.weightsHidden1Hidden2) + self.biasHidden2
        self.hidden2Output = np.tanh(self.hidden2Input)

        self.hidden3Input = np.dot(self.hidden2Output, self.weightsHidden2Hidden3) + self.biasHidden3
        self.hidden3Output = np.tanh(self.hidden3Input)

        self.outputInput = np.dot(self.hidden3Output, self.weightsHidden3Output) + self.biasOutput
        self.outputInput += sum(X[-64:]) # sum of original board inputs
        finalOutput = np.tanh(self.outputInput)
        return finalOutput


    def evaluate(self, board):
        subsquares = getSubsquares(board)
        #print(subsquares)
        subsquares = np.array(subsquares, dtype=np.float32)  # Ensure NumPy array
        return self.forward(subsquares, board)
    

    def createOffspring(self):
        print("initializing network...")        
        offspring = copy.deepcopy(self)
        offspring.sigma *= exp(tau*random.gauss(0,1))

        def mutate(l):
            return [wj + offspring.sigma * random.gauss(0,1) for wj in l]
        
        offspring.weightsInputHidden1 = mutate(offspring.weightsInputHidden1)
        offspring.biasHidden1 = mutate(offspring.biasHidden1)
        offspring.weightsHidden1Hidden2 = mutate(offspring.weightsHidden1Hidden2)
        offspring.biasHidden2 = mutate(offspring.biasHidden2)
        offspring.weightsHidden2Hidden3 = mutate(offspring.weightsHidden2Hidden3)
        offspring.biasHidden3 = mutate(offspring.biasHidden3)
        offspring.weightsHidden3Output = mutate(offspring.weightsHidden3Output)
        offspring.biasOutput = mutate(offspring.biasOutput)

        return offspring

if __name__ == "__main__":
    import chess
    board = chess.Board()
    print(getSubsquares(board))
