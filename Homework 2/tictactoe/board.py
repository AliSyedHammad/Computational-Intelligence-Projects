#The base code has been taken from the book "Invent Your Own Computer Games with Python" - Chapter 10
#available online at https://inventwithpython.com/chapters/index.html#chapter10.
#The base code has been modified to be used during a class exercise. 


# -*- coding: cp1252 -*-
import random

#from tictactoe import *

def drawBoard(board):
    print('     |   |')
    print('   ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('     |   |')
    print('---------------')
    print('     |   |')
    print('   ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('     |   |')
    print('---------------')
    print('     |   |')
    print('   ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('     |   |')

def playAgain():
    print('Do you want to play again? (yes or no)')
    another =input()
    return another.lower().startswith('y')

def makeMove(board, letter, move):
    board[move] = letter

def isWinner(bo, le):
# Given a board and a player’s letter, this function returns True if that player has won.
# We use bo instead of board and le instead of letter so we don’t have to type as much.
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal

def getBoardCopy(board):
    # Make a duplicate of the board list and return it the duplicate.
    dupeBoard = []
    for i in board:
        dupeBoard.append(i)
    return dupeBoard

def isSpaceFree(board, move):
    # Return true if the passed move is free on the passed board.
    return board[move] == ' '
 

def chooseRandomMoveFromList(board, movesList):
    # Returns a valid move from the passed list on the passed board.
    # Returns None if there is no valid move.
    possibleMoves = []
    
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)
            
    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None

def whoGoesFirst():
    if random.randint(0, 1) == 0:
        return 'Player1'
    else:
        return 'Player2'

def isBoardFull(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True

 
def Play(playe1Strategy, player2Strategy, visibleMode=True):
    theBoard = [' '] * 10
    player1Letter, player2Letter = ['X','O']
    turn = whoGoesFirst()
    # print('The ' + turn + ' will go first.')
    gameIsPlaying = True
    i = 0
    while gameIsPlaying:
        #print('turn:'+turn)
        if turn == 'Player1':
            # Player 1’s turn.
            if (visibleMode):
                # print("Player 1's move...")
                drawBoard(theBoard)
                
            move = playe1Strategy(theBoard,player1Letter)
            makeMove(theBoard, player1Letter, move)
            if isWinner(theBoard, player1Letter):
                if (visibleMode):
                    drawBoard(theBoard)
                # print('Player 1 has won the game!')
                return 1
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    if visibleMode:
                        drawBoard(theBoard)
                    # print('The game is a tie!')
                    return 0
                    break
                else:
                    turn = 'Player2'
        else:
            # Player 2’s turn.
            if (visibleMode):
                # print("Player 2's move...")
                drawBoard(theBoard)

            move = player2Strategy(theBoard, player2Letter)
            makeMove(theBoard, player2Letter, move)
            if isWinner(theBoard, player2Letter):
                if visibleMode:
                    drawBoard(theBoard)
                #print('Player 2 has won the game !')
                return -1
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    if visibleMode:
                        drawBoard(theBoard)
                    # print('The game is a tie!')
                    return 0
                
                    break
                else:
                    turn = 'Player1'

def PlayMultiple(noofGames,player1Strategy, player2Strategy, visibleMode=False):
    gamesWon = 0
    gamesTie = 0
    gamesLost = 0

    for i in range(1,noofGames+1):
        result = Play(player1Strategy, player2Strategy,visibleMode)
        if result == 1:
            gamesWon +=1
        elif result == 0:
            gamesTie +=1
        else:
            gamesLost +=1
    
    return (gamesWon,gamesLost, gamesTie)