import torch
import torch.nn as nn
from torch import jit

class ProximalPolicyOptimization(nn.Module):
    name = "ProximalPolicyOptimization"
    
    def __init__(self, input_size, output_size, hidden_layers=2, hidden_size=64, activation=nn.ReLU):
        super(ProximalPolicyOptimization, self).__init__()
        self.layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        policy_layers = [nn.Linear(input_size, hidden_size), activation()]
        for _ in range(hidden_layers - 1):
            policy_layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        policy_layers.append(nn.Linear(hidden_size, output_size))
        self.policy_model = nn.Sequential(*policy_layers)
        
        value_layers = [nn.Linear(input_size, hidden_size), activation()]
        for _ in range(hidden_layers - 1):
            value_layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        value_layers.append(nn.Linear(hidden_size, 1))
        self.value_model = nn.Sequential(*value_layers)
    
    def policy(self, observation):
        logits = self.policy_model(observation)
        return torch.softmax(logits, dim=-1)

    def value(self, observation):
        return self.value_model(observation).squeeze(-1)

    def act(self, observation, action_mask=None):
        action_probs = self.policy(observation)
        
        if action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.float32)
            action_probs = action_probs * action_mask
            action_probs = action_probs / action_probs.sum()
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_prob = action_probs[action.item()].detach()
        return action.item(), action_prob
    
    def compute_loss(self, observations, actions, old_action_probs, returns, advantages, clip_epsilon=0.2):
        action_probs = self.policy(observations).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        values = self.value(observations).squeeze(-1)

        ratio = action_probs / old_action_probs
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        policy_loss = - torch.min(ratio * advantages, clipped_ratio * advantages).mean() 
        value_loss = 0.5 * (returns - values).pow(2).mean()

        # Entropy bonus
        entropy = - (action_probs * torch.log(action_probs + 1e-8)).mean()
        entropy_coef = 0.01  # Adjust as needed
        policy_loss -= entropy_coef * entropy
        
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss 