import torch
import torch.nn as nn

class BasicPolicyGradient(nn.Module):
    path = "BasicPolicyGradient.py"
    
    def __init__(self, input_size, output_size, hidden_layers=2, hidden_size=64, activation=nn.ReLU):
        super(BasicPolicyGradient, self).__init__()
        if hidden_layers == 0:
            self.model = nn.Sequential(nn.Linear(input_size, output_size))
            return
        
        layers_list = [nn.Linear(input_size, hidden_size), activation()]
        for _ in range(hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_size, hidden_size))
            layers_list.append(activation())
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
        return action, action_probs

    def compute_loss(self, actions, action_probs, rewards, total_steps):
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        return - log_probs * rewards / total_steps
    