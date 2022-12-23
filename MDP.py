# Parking Model MDP
from sklearn import kernel_ridge
import numpy as np
import random

class BanditMDP:
    """
    Paramters:
        train_data - tuple of 2D array containing X and Y training data
               in their corresponding cluster
        train_data - tuple of array containing X and Y testing data
        max_bins - list of the total data points for each cluster
    """
    def __init__(self, train_data, test_data, max_bins):
        lambda_val = 0.1
        gamma = 0.001
        self.rrg = kernel_ridge.KernelRidge(alpha=lambda_val, kernel='rbf', gamma=gamma)
        self.bins_size = len(max_bins)
        self.max_bins = max_bins
        # empty intital state
        self.state = BanditState([0]*self.bins_size)
        # initialize data
        self.X_train = train_data[0]
        self.Y_train = train_data[1]
        self.X_test = test_data[0]
        self.Y_test = test_data[1]
        # get possible operators based on bins
        self.operators = [0]
        for i in range(self.bins_size):
            self.operators.append(i+1)
            self.operators.append(-(i+1))
        # Initialize Q-Table and Policy
        self.policy = {}
        self.q_table = {}

        # MDP Parameters
        self.alpha = 0.5
        self.epsilon = 0.5
        self.rng = random.Random()
        
    """
    Get the training data corresponding to current state
    Return:
        x - training data input
        y - training data output
    """
    def get_data(self, curr_state):
        x = []
        y = []
        for i in range(self.bins_size):
            count = self.state.bins[i]
            if count != 0:
                for j in range(count):
                    x.append(self.X_train[i][j])
                    y.append(self.Y_train[i][j])
                
        return np.array(x),np.array(y)
            
    def transition(self, action):
        prev_state = self.state.copy()
        self.state.move(action, self.max_bins)
        return prev_state
        
    def reward(self, curr_state):
        x_train, y_train = self.get_data(curr_state)
        print(x_train)
        if x_train.size == 0:
            return 0
        self.rrg.fit(x_train, y_train)
        y_pred = self.rrg.predict(self.X_test)
        mae = np.mean(np.abs(self.Y_test-y_pred))
        print('Reward:', mae)
        return mae

    """
    Based on the q table determine the optimal policy for each state
    """
    def get_policy(self):
        for key in self.q_table:
            value = self.q_table[key]
            index = np.argmax(np.array(value))
            self.policy[key] = self.operators[index]

    def q_update(self, action, prev_state):
        max_q_prime = 0
        if action == 0:
            max_q_prime = self.reward(prev_state)
        elif self.state in self.q_table.keys():
            max_q_prime = max(self.q_table[self.state])
        else:
            self.q_table[self.state] = [0]*len(self.operators)
            
        sample = max_q_prime
        if prev_state not in self.q_table.keys():
            self.q_table[prev_state] = [0]*len(self.operators)
        self.q_table[prev_state][abs(action)-1] = (1-self.alpha)*self.q_table[prev_state][abs(action)-1] + self.alpha*sample

    def get_move(self):
        coin = self.rng.random()
        if coin >= self.epsilon:
            index = np.argmax(np.array(self.q_table[self.state]))
            return self.operators[index]
        else:
            return self.rng.choice(self.operators)
        
    def run_iteration(self):
        if self.state not in self.q_table.keys():
            self.q_table[self.state] = [0]*len(self.operators)
        curr_move = self.get_move()
        print(curr_move)
        prev_state = self.transition(curr_move)
        self.q_update(curr_move, prev_state)
        
class BanditState:
    """
    State represent the current Parking-Model
    Parameter:
        bins - an array with count of the number of sample for each clusters
    """
    def __init__(self, bins):
        self.bins = bins

    def copy(self):
        rep = self.bins.copy()
        return BanditState(rep)

    """
    Transition state to another given an action
    Parameters:
        action - valid operators
        max_bins - max count of sample in each bin
    Return:
        integer 1 if the action is valid otherwise -1
    """
    def move(self, action, max_bins):
        if action == 0:
            return
        index = abs(action) - 1
        if action < 0 and self.bins[index] != 0:
            self.bins[index] -= 1
        elif self.bins[index] != max_bins[index]:
            self.bins[index] += 1
        print(self.bins)

    
    def __eq__(self, other):
        return self.bins == other.bins
    
    def __hash__(self):
        return sum([self.bins[i]*(10**i) for i in range(len(self.bins))])
