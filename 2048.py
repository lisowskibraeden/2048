import pygame
import math
from random import randint, seed
import random
from datetime import datetime
from copy import deepcopy
import argparse
import numpy as np
from ai import DQNAgent
import torch.optim as optim
import torch 
import distutils.util
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
DEVICE = 'cuda'

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

clock = pygame.time.Clock()
class game:
    def __init__(self):
        self.board = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
        self.game_over = False
        self.score = 0
        game.randomAdd(self)
        game.randomAdd(self)

    def restart(self):
        self.board = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
        self.score = 0
        self.game_over = False
        self.__init__(self)
        
    def checkMoves(self):
        testBoard = deepcopy(self.board)
        game.left(self, testBoard)
        if testBoard != self.board:
            return False
        game.right(self, testBoard)
        if testBoard != self.board:
            return False
        game.up(self, testBoard)
        if testBoard != self.board:
            return False
        game.down(self, testBoard)
        if testBoard != self.board:
            return False
        return True

    def checkBoard(self):
        if not any(0 in x for x in self.board) and game.checkMoves(self):
            self.game_over = True
            # game.restart(self)

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
                        self.score += board[y][i] + board[y][j]
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

def display(game, screen, myfont):
    screen.fill((255, 255, 255))
    score_text = myfont.render('Score', True, (0,0,0))
    score_width, score_height = myfont.size('Score')
    screen.blit(score_text, (-score_width / 2 + 250, -score_height / 2 + 50))
    score_num = myfont.render(str(game.score), True, (0,0,0))
    score_num_width, score_num_height = myfont.size(str(game.score))
    screen.blit(score_num, (-score_num_width / 2 + 250, -score_num_height / 2 + 100))
    for i in range(len(game.board)):
        for x in range(len(game.board)):
            num = myfont.render(str(game.board[i][x]), True, (0, 0, 0))
            num_width, num_height = myfont.size(str(game.board[i][x]))
            if game.board[i][x] != 0:
                pygame.draw.rect(screen, colors[int(math.log(game.board[i][x], 2))], pygame.Rect(50 + 500/5 * i, 250 + 500/5 * x, 100, 100), 0, 3)
                screen.blit(num, ((-num_width / 2) + 100 + 500/5 * i, (-num_height / 2) + 300 + 500/5 * x))
            else:
                pygame.draw.rect(screen, 'gray', pygame.Rect(50 + 500/5 * i, 250 + 500/5 * x, 100, 100), 0, 3)
    pygame.display.flip()

def graphics():
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
        while g.game_over == True:
            button = pygame.Rect(175, 150, 150, 50)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if button.collidepoint(mouse_pos):
                       g.game_over = False
                       g.restart()
            s = pygame.Surface((500,700))
            s.set_alpha(5)
            s.fill((255,255,255))
            screen.blit(s, (0,0))
            pygame.draw.rect(screen, [255, 0, 0], button)
            game_over_text = myfont.render('Game Over', True, (0,0,0))
            game_over_width, game_over_height = myfont.size('Game Over')
            screen.blit(game_over_text, (-game_over_width / 2 + 250, -game_over_height / 2 + 175))
            pygame.display.flip()
            clock.tick(30)
        screen.fill((255, 255, 255))
        score_text = myfont.render('Score', True, (0,0,0))
        score_width, score_height = myfont.size('Score')
        screen.blit(score_text, (-score_width / 2 + 250, -score_height / 2 + 50))
        score_num = myfont.render(str(g.score), True, (0,0,0))
        score_num_width, score_num_height = myfont.size(str(g.score))
        screen.blit(score_num, (-score_num_width / 2 + 250, -score_num_height / 2 + 100))
        for i in range(len(g.board)):
            for x in range(len(g.board)):
                num = myfont.render(str(g.board[i][x]), True, (0, 0, 0))
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
        clock.tick(30)
    pygame.quit()

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record

def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()

def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    pygame.font.init() 
    myfont = pygame.font.SysFont('Arial', 30)
    logo = pygame.image.load("logo.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("2048")
    screen = pygame.display.set_mode([500, 700])
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        g = game()

        # Perform first move
        
        steps = 0       # steps since the last positive reward
        while (not g.game_over) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(g)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(4)[randint(0,3)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 16)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(4)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            if final_move[0] == 1:
                g.move(0)
            elif final_move[1] == 1:
                g.move(1)
            elif final_move[2] == 1:
                g.move(2)
            elif final_move[3] == 1:
                g.move(3)
            state_new = agent.get_state(g)
            # set reward for the new state
            reward = agent.set_reward(g.score, g.game_over)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, g.game_over)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, g.game_over)

            # record = get_record(g.score, record)
            # display(g, screen, myfont)
            # pygame.time.wait(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += g.score
        print(f'Game {counter_games}      Score: {g.score}')
        score_plot.append(g.score)
        counter_plot.append(counter_games)
    mean, stdev = statistics.mean(score_plot), statistics.stdev(score_plot)  
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 1
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 100
    params['memory_size'] = 10000
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = True
    params["test"] = False
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params

if __name__ == "__main__":
    pygame.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = 1
    if params['train']:
        print("Training...")
        params['load_weights'] = True   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)
