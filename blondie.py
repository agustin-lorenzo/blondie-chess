import network
import chess
import copy
import random
import math
import pickle
import os
import pandas as pd

class Blondie(network.Network):

    def minimax(self, board, player, depth=4):
        if depth == 0:
            return self.evaluate(board), None

        bestMove = None
        bestScore = math.inf * -player # maximizing for white (player = 1), minimizing for black (player = 0)
        
        for move in board.legal_moves:
            boardCopy = copy.deepcopy(board)
            boardCopy.push(move)
            score, _ = self.minimax(boardCopy, -player, depth-1)
            if player == 1: # white
                if score > bestScore:
                    bestScore = score
                    bestMove = move
            else:
                if score < bestScore:
                    bestScore = score
                    bestMove = move
        if bestMove == None:
            print("Board: ", board)
            print("Player: ", player)
        return bestScore, bestMove
                    

            

    def alphabeta(self, board, player, depth=7, alpha=-math.inf, beta=math.inf):
        # TODO: adapt previous alphabeta function for chess library
        return



def playGame(player1, player2, depth, alphabeta=False):
    board = chess.Board()
    
    # Randomly choose which player starts
    current_player = -1
    
    while not (board.is_checkmate()
               or board.is_stalemate()
               or board.is_insufficient_material()):
        if current_player == 1:
            if alphabeta:
                _, player1Move = player1.alphabeta(board, 1, depth)
            else:
                _, player1Move = player1.minimax(board, 1, depth)
            board.push(player1Move)
            if  (board.is_checkmate()
                 or board.is_stalemate()
                 or board.is_insufficient_material()):
                return 1
            current_player = -1
            
        else:
            if alphabeta:
                _, player2Move = player2.alphabeta(board, -1, depth)
            else:
                _, player2Move = player2.minimax(board, -1, depth)
            board.push(player2Move)
            if  (board.is_checkmate()
                 or board.is_stalemate()
                 or board.is_insufficient_material()):
                return 0
            if  (board.is_checkmate()
                 or board.is_stalemate()
                 or board.is_insufficient_material()):
                return -1
            current_player = 1
        os.system('clear')
        print(board)
    return 0


def runES(generations = 840, depth = 7, alphabeta=True):
    networks = [Blondie() for _ in range(15)]
    offspring = [network.createOffspring() for network in networks]
    networks.extend(offspring)

    averages = []

    for i in range(generations):
        print(f"Generation: {i+1}/{generations}; Depth = {depth+1}")

        j = 1
        totalNetFitness = 0
        for currentNetwork in networks:
            
            possibleOpponents = [n for n in networks if n is not currentNetwork]
            selctedOpponents = random.choices(possibleOpponents, k=5)

            for opponent in selctedOpponents: # play each of 5 selected opponents to determine current network's fitness
                
                outcome = playGame(currentNetwork, opponent, depth, False)
                if outcome == 1: currentNetwork.fitness += 1
                elif outcome == -1: currentNetwork.fitness -= 2

            print(f"net {j} fitness: {currentNetwork.fitness}")
            j += 1
            totalNetFitness += currentNetwork.fitness
        averageFitness = totalNetFitness/30
        averages.append(averageFitness)
        print("AVERAGE FITNESS FOR GENERATION: ", averageFitness)
        print("===========================")
            
        networks.sort(key=lambda g: g.fitness, reverse=True)
        networks = networks[:15] # select survivors
        for network in networks: network.fitness = 0 # reset fitness for new round
        offspring = [network.createOffspring() for network in networks] # create offspring from selected individuals
        networks.extend(offspring)

    # dataframe to plot fitness over time
    fitnessDf = pd.DataFrame({'averageFitness': averages})
    fitnessDf.to_csv(f'{generations}g{depth+1}dfitnessOverTime.csv')
    
    networks.sort(key=lambda g: g.fitness)
    return networks # save top 15 networks

if __name__ == "__main__":
    bestNetworks = runES(840, depth=2)
    with open('840g4dbest15Networks.pkl', 'wb') as f:
        pickle.dump(bestNetworks, f) 

