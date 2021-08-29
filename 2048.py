import pygame
import math
from random import randint, seed
from datetime import datetime
from copy import deepcopy

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

seed(datetime.now().timestamp())

newVal = [2, 2, 2, 2, 2, 2, 2, 2, 2, 4]
colors = [0, '#eeb17d', '#f29866', '#f07f61','#f46042','#eace75', '#edcb67','#ecc85a', '#e5c254','#ecba4e', '#ffae00', '#bbfaa2', '#7bff47', '#48ff00', '#c675ff', '#ac38ff', '#000000']
score = 0
class game:
    board = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    def __init__(self):
        game.randomAdd(self)
        game.randomAdd(self)

    def restart(self):
        game.printBoard(self, game.board) # game restarts too early
        game.board = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
        score = 0
        game.__init__(self)
        
    def checkMoves(self):
        testBoard = deepcopy(game.board)
        game.left(self, testBoard)
        if testBoard != game.board:
            return False
        game.right(self, testBoard)
        if testBoard != game.board:
            return False
        game.up(self, testBoard)
        if testBoard != game.board:
            return False
        game.down(self, testBoard)
        if testBoard != game.board:
            return False
        return True

    def checkBoard(self):
        if not any(0 in x for x in game.board) and not game.checkMoves(self):
            game.restart(self)

    def randomAdd(self):
        x = randint(0, 3)
        y = randint(0, 3)
        while self.board[y][x] != 0:
            x = randint(0, 3)
            y = randint(0, 3)
        v = randint(0, 9)
        self.board[y][x] = newVal[v]

    def printBoard(self, board):
        print(board)
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                print (board[i][j], end = ' ')
            print ("")
        print()

    def move(self, num):
        state = 0
        if num == 0:
            state = game.left(self, self.board)
        elif num == 1:
            state = game.right(self, self.board)
        elif num == 2:
            state = game.down(self, self.board)
        elif num == 3: 
            state = game.up(self, self.board)
        if state == 0:
            game.randomAdd(self)
            game.checkBoard(self)
        
    def up(self, board):
        global score
        prev = deepcopy(board)
        for y in range(0, 4):
            # Merge
            for i in range(0, len(board[y])):
                if board[y][i] == 0:
                    continue
                for j in range(i + 1, len(board[y])):
                    if board[y][j] == 0:
                        continue
                    elif board[y][i] == board[y][j]:
                        score += board[y][i] + board[y][j]
                        board[y][i] = board[y][i] + board[y][j]
                        board[y][j] = 0
                        i = j
                        break
                    else:
                        break
            # Move
            for i in range(0, len(board[y])):
                if board[y][i] != 0:
                    num = board[y][i]
                    for j in range(i, -1, -1):
                        if board[y][j] == 0:
                            board[y][j] = num
                            board[y][j + 1] = 0
        if prev == board:
            return -1
        else:
            return 0
    def left(self, board):
        game.rotate(self, board)
        num = game.up(self, board)
        game.rotate(self, board)
        game.rotate(self, board)
        game.rotate(self, board)
        return num
    def down(self, board):
        game.rotate(self, board)
        game.rotate(self, board)
        num = game.up(self, board)
        game.rotate(self, board)
        game.rotate(self, board)
        return num
    def right(self, board): 
        game.rotate(self, board)
        game.rotate(self, board)
        game.rotate(self, board)
        num = game.up(self, board)
        game.rotate(self, board)
        return num
    # From https://www.geeksforgeeks.org/inplace-rotate-square-matrix-by-90-degrees/
    def rotate(self, board):
        for x in range(0, int(len(board) / 2)):
         
        # Consider elements in group  
        # of 4 in current square
            for y in range(x, len(board)-x-1):
                
                # store current cell in temp variable
                temp = board[x][y]
    
                # move values from right to top
                board[x][y] = board[y][len(board)-1-x]
    
                # move values from bottom to right
                board[y][len(board)-1-x] = board[len(board)-1-x][len(board)-1-y]
    
                # move values from left to bottom
                board[len(board)-1-x][len(board)-1-y] = board[len(board)-1-y][x]
    
                # assign temp to left
                board[len(board)-1-y][x] = temp

def main():
    pygame.init()
    logo = pygame.image.load("logo.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("2048")
    screen = pygame.display.set_mode([500, 700])
    pygame.font.init() 
    myfont = pygame.font.SysFont('Arial', 30)
    running = True
    g = game()
    screen.fill((255, 255, 255))
    while running:
        score_text = myfont.render('Score', False, (0,0,0))
        score_width, score_height = myfont.size('Score')
        screen.blit(score_text, (-score_width / 2 + 250, -score_height / 2 + 50))
        score_num = myfont.render(str(score), False, (0,0,0))
        score_num_width, score_num_height = myfont.size(str(score))
        screen.blit(score_num, (-score_num_width / 2 + 250, -score_num_height / 2 + 100))
        for i in range(len(g.board)):
            for x in range(len(g.board)):
                num = myfont.render(str(g.board[i][x]), False, (0, 0, 0))
                num_width, num_height = myfont.size(str(g.board[i][x]))
                if g.board[i][x] != 0:
                    pygame.draw.rect(screen, colors[int(math.log(g.board[i][x], 2))], pygame.Rect(50 + 500/5 * i, 250 + 500/5 * x, 100, 100), 0, 3)
                    screen.blit(num, ((-num_width / 2) + 100 + 500/5 * i, (-num_height / 2) + 300 + 500/5 * x))
                else:
                    pygame.draw.rect(screen, 'gray', pygame.Rect(50 + 500/5 * i, 250 + 500/5 * x, 100, 100), 0, 3)
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    g.move(1)
                elif event.key == K_DOWN:
                    g.move(2)
                elif event.key == K_LEFT:
                    g.move(0)
                elif event.key == K_UP:
                    g.move(3)
                elif event.key == K_ESCAPE:
                    running = False
            elif event.type == QUIT:
                running = False
        pygame.display.flip()
        screen.fill((255, 255, 255))
    pygame.quit()

if __name__ == "__main__":
    main()
