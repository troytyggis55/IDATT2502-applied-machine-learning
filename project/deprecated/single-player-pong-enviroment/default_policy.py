from random import random

import torch
import torch.nn as nn


class DefaultPolicyAgent(nn.Module):
    def __init__(self, layers=2, hidden_size=64, input_size=5, output_size=3):        
        super(DefaultPolicyAgent, self).__init__()
        if layers == 0:
            self.model = nn.Sequential(nn.Linear(input_size, output_size))
            return
        
        layers_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden_size, hidden_size))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers_list)
    def policy(self, observation):
        logits = self.model(observation)
        action_probs = torch.softmax(logits, dim=0)
        return action_probs

    def act(self, observation, action_mask=None):
        action_probs = self.policy(observation)
        if action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.float32)
            action_probs = action_probs * action_mask
            action_probs = action_probs / action_probs.sum()
        action = torch.multinomial(action_probs, 1).item()
        return action, torch.log(action_probs[action])

    def loss(self, log_probs, rewards, total_steps):
        return - torch.tensor(sum(log_probs) * rewards / total_steps, requires_grad=True)  
    