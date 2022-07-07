# -*- coding: utf-8 -*-
import random
# from neural_network import NeuralNetwork
from board import *

def getPlayerMove(board, playerLetter):
    # Let the player type in their move.
    move = ''
    possibleMoves = [1,2,3,4,5,6,7,8,9]
    while move not in possibleMoves or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = int(input())

    return int(move)

def getNaiveMove(board, playerLetter):
    # Given a board and a letter, determine where to move and return that move.
    if playerLetter == 'X':
        opponentLetter = 'O'
    else:
        opponentLetter = 'X'

    # check if we can win in the next move
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i
    
    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [1,2,3, 4,5, 6,7, 8,9])

def getRandomMove(board, playerLetter):
    move = chooseRandomMoveFromList(board, [1,2,3,4,5,6,7,8,9])
    return move

def getIntelligentMove(board, playerLetter):
    # Given a board and the letter, determine where to move and return that move.
    if playerLetter == 'X':
        opponentLetter = 'O'
    else:
        opponentLetter = 'X'

    # Here is our algorithm for  Tic Tac Toe AI:
    # First, check if we can win in the next move
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i
    
    # Check if the opponent could win on their next move, and block them.
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, opponentLetter, i)
            if isWinner(copy, opponentLetter):
                return i

    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 5):
        return 5

    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [2, 4, 6, 8])
#---Emad----------------------------------------------------------

def getUnintelligentMove(board, playerLetter):
    for pos in range(1,10):
        boardCopy= getBoardCopy(board)
        if isSpaceFree(boardCopy,pos):
            return pos
def getCornerMove(board,playerLetter):
    for pos in [1,3,7,9,5]:
        boardCopy= getBoardCopy(board)
        if isSpaceFree(boardCopy,pos):
            return pos
    return getUnintelligentMove(boardCopy,playerLetter)

#--------------------------------------------------------------


# def getMove(board, playerLetter):
#     model = NeuralNetwork(9, 14, 9)
#     nextMove = model.predict(inputs=encodedBoard(board))
#     print(nextMove)
#     return nextMove

#--------------------------------------------------------------
# print('Welcome to Tic Tac Toe!')
#Play(getPlayerMove, getPlayerMove)        #Two key-board players game
# (W,L,T) = PlayMultiple(500,getMove,getIntelligentMove)  #Simulate 500 matches between Naive and AI agent
# print ('\nTotal games Played:\t{0}\nGames Won by Player 1\t:{1}\nGames Won by Player 2\t:{2}\nGames Tie:{3}'.format(500,W, L,T))

#Play(getPlayerMove,getNaiveMove)

