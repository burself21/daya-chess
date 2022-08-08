import numpy as np
import pandas as pd
from .model import FFNN, CNN
#from main2 import FFNN
import torch
from copy import copy
from torch.utils.data import DataLoader, Dataset
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN

class RandomAgent:
    def __init__(self, seed):
        if seed:
            np.random.seed(seed)
    
    def get_action(self, env):
        return np.random.choice(env.possible_actions)

class RLAgent:
    def __init__(self, model_file):
        here = os.path.dirname(os.path.abspath(__file__))
        self.model = DQN.load(os.path.join(here, model_file))
    
    def get_action(self, env):
        action = self.model.predict(env.state, deterministic=False)[0]
        if action not in env.possible_actions:
            action = min(env.possible_actions, key=lambda x: abs(action - x))
        return action

class ChessAgent:
    #[1024, 512, 256, 128, 64]
    #[1024, 512, 128]
    def __init__(self, model_type, model_file, max_depth=1, max_breadth=3):
        # get model file path
        here = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(here, model_file)
        # hidden_sizes=[1024, 512, 256, 128, 64]
        #[2048, 1024, 512, 256, 128, 64]
        #[1024, 1024, 1024, 1024]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        if model_type == "FFNN":
            hidden_sizes = [1024, 512, 256, 128, 64]
            self.model = FFNN(32, hidden_dims=hidden_sizes).to(self.device)
        else:
            self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(filename)['model_state_dict'])
        self.model.eval()
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.type = model_type
    
    def process_position(self, env):
        if self.type == "FFNN":
            return np.append(env.state.reshape(-1), np.array([env.current_player_is_white, env.white_king_castle_possible, env.white_queen_castle_possible, 
                                                    env.black_king_castle_possible, env.black_queen_castle_possible], dtype=np.int8))

        data = env.state.reshape(1, 8, 8)
        data = np.concatenate([data, (data > 0).astype(int)], axis=0)
        data = np.concatenate([data, [[[int(env.current_player_is_white)] * 8] * 8]], axis=0)
        data[0] = np.abs(data[0])
        return data

    def get_best_action(self, env):
        possible_actions = env.possible_actions
        positions = []

        for action in possible_actions:
            env_copy = env.copy()
            env_copy.step(action)
            positions.append(self.process_position(env_copy))
        

        dataset = torch.FloatTensor(np.array(positions)).to(self.device)
        with torch.no_grad():
            evals = torch.reshape(self.model(dataset), (-1,))

        for i in range(len(possible_actions)):
            print(env.move_to_string(env.action_to_move(possible_actions[i])) + ":", evals[i].item())

        if env.current_player_is_white:
            return possible_actions[torch.argmax(evals).item()]
        else:
            return possible_actions[torch.argmin(evals).item()]

    def get_best_action_eval(self, env):
        possible_actions = env.possible_actions
        positions = []

        for action in possible_actions:
            env_copy = env.copy()
            env_copy.step(action)
            positions.append(self.process_position(env_copy))
        

        dataset = torch.FloatTensor(np.array(positions)).to(self.device)
        with torch.no_grad():
            evals = torch.reshape(self.model(dataset), (-1,)) 
        
        for i in range(len(possible_actions)):
            print(env.move_to_string(env.action_to_move(possible_actions[i])) + ":", evals[i].item())

        if env.current_player_is_white:
            return torch.max(evals).item()
        else:
            return torch.min(evals).item()

    # evaluate position by search and minmaxing
    def evaluate_position(self, env, depth, breadth):
        if env.done:
            if env.black_won:
                return -100
            if env.white_won:
                return 100
            else:
                return 0

        if depth == 1:
            return self.get_best_action_eval(env)
        
        possible_actions = env.possible_actions
        
        positions = []
        environments = []
        
        for action in possible_actions:
            env_copy = env.copy()
            env_copy.step(action)
            positions.append(self.process_position(env_copy))
            environments.append(env_copy)

        dataset = torch.FloatTensor(np.array(np.array(positions))).to(self.device)

        with torch.no_grad():
            evals = torch.reshape(self.model(dataset), (-1,))
        
        for i in range(len(possible_actions)):
            print(env.move_to_string(env.action_to_move(possible_actions[i])) + ":", evals[i].item())

        if env.current_player_is_white:
            if len(possible_actions) > breadth:
                top_vals, top_indices = torch.topk(evals, breadth, sorted=False)
                indices = top_indices[(top_vals - torch.max(top_vals)) > -1.5].cpu()
                #environments = np.array(environments)[top_indices[(top_vals - torch.max(top_vals)) > -1.5].cpu()]
                #environments = np.array(environments)[torch.topk(evals, breadth).indices.cpu()]
            evaluation = max([self.evaluate_position(environments[idx], depth - 1, breadth) for idx in indices])
            print("evaluation:", evaluation)
            return evaluation
        else:
            if len(possible_actions) > breadth:
                bot_vals, bot_indices = torch.topk(evals, breadth, sorted=False, largest=False)
                indices = bot_indices[(bot_vals - torch.min(bot_vals)) < 1.5].cpu()
                #environments = np.array(environments)[bot_indices[(bot_vals - torch.min(bot_vals)) < 1.5].cpu()]
                #environments = np.array(environments)[torch.topk(evals, breadth, largest=False).indices.cpu()]
            evaluation = min([self.evaluate_position(environments[idx], depth - 1, breadth) for idx in indices])
            print("evaluation:", evaluation)
            return evaluation


    # give a prediction given a ChessEnv
    def get_action(self, env):
        
        if self.max_depth == 1:
            return self.get_best_action(env)

        possible_actions = env.possible_actions
        evaluations = []
        positions = []
        environments = []

        for action in possible_actions:
            print(env.move_to_string(env.action_to_move(action)))
            print("____________________________")
            env_copy = env.copy()
            env_copy.step(action)
            positions.append(self.process_position(env_copy))
            environments.append(env_copy)
            
        dataset = torch.FloatTensor(np.array(np.array(positions))).to(self.device)

        with torch.no_grad():
            evals = torch.reshape(self.model(dataset), (-1,))
        
        if env.current_player_is_white:
            if len(possible_actions) >= self.max_breadth:
                top_vals, top_indices = torch.topk(evals, self.max_breadth, sorted=False)
                indices = top_indices[(top_vals - torch.max(top_vals)) > -1.5].cpu()
            else:
                indices = range(len(possible_actions))
                #environments = np.array(environments)[top_indices[(top_vals - torch.max(top_vals)) > -1.5].cpu()]
                #environments = np.array(environments)[torch.topk(evals, breadth).indices.cpu()]
            return possible_actions[np.argmax([self.evaluate_position(environments[idx], self.max_depth - 1, self.max_breadth) for idx in indices])]
        else:
            if len(possible_actions) >= self.max_breadth:
                bot_vals, bot_indices = torch.topk(evals, self.max_breadth, sorted=False, largest=False)
                indices = bot_indices[(bot_vals - torch.min(bot_vals)) < 1.5].cpu()
            else:
                indices = range(len(possible_actions))
                #environments = np.array(environments)[bot_indices[(bot_vals - torch.min(bot_vals)) < 1.5].cpu()]
                #environments = np.array(environments)[torch.topk(evals, breadth, largest=False).indices.cpu()]
            return possible_actions[np.argmin([self.evaluate_position(environments[idx], self.max_depth - 1, self.max_breadth) for idx in indices])]
        
        #evaluations = [self.evaluate_position(env_copy, self.max_depth - 1, self.max_breadth) for ]
        
        # if env.current_player_is_white:
        #     return possible_actions[np.argmax(evaluations)]
        # else:
        #     return possible_actions[np.argmin(evaluations)]



            
