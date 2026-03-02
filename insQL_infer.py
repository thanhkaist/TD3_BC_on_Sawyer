# Load standalone actor and critic model for inference
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TASK = 'sawyer-pick-lift-banana-v0'

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TwoStepMLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16, r_dim=16):

        super(TwoStepMLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        self.r_mlp = nn.Sequential(
            SinusoidalPosEmb(r_dim),
            nn.Linear(r_dim, r_dim * 2),
            nn.Mish(),
            nn.Linear(r_dim * 2, r_dim),
        )

        input_dim = state_dim + action_dim + t_dim + r_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, r_time, state):

        t = self.time_mlp(time)
        r = self.r_mlp(r_time)
        x = torch.cat([x, t, r, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dim = 10
action_dim = 5
max_action = np.array([1.0] * action_dim)  # Assuming max_action is an array of 1.0 for each action dimension
model = TwoStepMLP(state_dim=state_dim, action_dim=action_dim, device=device)
# actor = InstantFlow(state_dim=state_dim, action_dim=action_dim, model=model, max_action=max_action).to(device)
critic = Critic(state_dim, action_dim).to(device)

if TASK == 'sawyer-pick-lift-banana-v0':
    model_load_path = './models/sawyer-pick-lift-banana-v0/actor.pth'
    critic_load_path = './models/sawyer-pick-lift-banana-v0/critic.pth'
elif TASK == 'sawyer-move-box-v0':
    model_load_path = './models/sawyer-move-box-v0/actor.pth'
    critic_load_path = './models/sawyer-move-box-v0/critic.pth'
elif TASK == 'sawyer-open-drawer-v0':
    model_load_path = './models/sawyer-open-drawer-v0/actor.pth'
    critic_load_path = './models/sawyer-open-drawer-v0/critic.pth'
# actor.load_state_dict(torch.load(model_load_path))
# Load actor weights and extract only the TwoStepMLP model weights
actor_state_dict = torch.load(model_load_path)
# Extract only the model weights (keys that start with 'model.')
model_state_dict = {k.replace('model.', ''): v for k, v in actor_state_dict.items() if k.startswith('model.')}
model.load_state_dict(model_state_dict)
model.to(device)


critic.load_state_dict(torch.load(critic_load_path))

print(f'Model loaded from {model_load_path} and {critic_load_path}')
# sampling action from the loaded policy
state = np.random.randn(state_dim)
state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
# action = actor.sample(state_tensor).cpu().data.numpy().flatten()


def sample_action(model, state_tensor):
    batch_size = state_tensor.shape[0]
    t_time = torch.full((batch_size,), 1, device=device, dtype=torch.long)
    r_time = torch.full((batch_size,), 0, device=device, dtype=torch.long)
    noise = torch.randn((batch_size, action_dim)).to(device)
    with torch.no_grad():
        action = noise - model(noise, t_time, r_time, state_tensor)
    return action

def sample_action_with_q(model, critic, state_tensor, num_samples=50):
    aug_state_tensor = torch.repeat_interleave(state_tensor, repeats=num_samples, dim=0)
    batch_size = aug_state_tensor.shape[0]

    t_time = torch.full((batch_size,), 1, device=device, dtype=torch.long)
    r_time = torch.full((batch_size,), 0, device=device, dtype=torch.long)
    noise = torch.randn((num_samples, action_dim)).to(device)
    with torch.no_grad():
        actions = noise - model(noise, t_time, r_time, aug_state_tensor)
        q_values = critic.q_min(aug_state_tensor, actions).flatten()
        idx = torch.multinomial(F.softmax(q_values, dim=0), 1)
    return actions[idx]


# action = sample_action(model, state_tensor).cpu().data.numpy().flatten()
# print(f"Sampled action: {action}")

action_with_q = sample_action_with_q(model, critic, state_tensor).cpu().data.numpy().flatten()
print(f"Sampled action with Q: {action_with_q}")
