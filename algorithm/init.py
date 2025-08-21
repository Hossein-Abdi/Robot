
import numpy as np
import torch
import torch.nn as nn
# import torchrl
import os
import sys
sys.path.append("algorithm")
from algorithm.init import *
import algorithm.memory as memory
import algorithm.models
# from algorithm.decision_transformer.models.decision_transformer import DecisionTransformer
from transformers import DecisionTransformerConfig, DecisionTransformerModel







device = torch.device('cuda')
dtype = torch.float32
result_dir = "/mnt/iusers01/fatpou01/compsci01/x47085ha/scratch/result/"

BATCH_SIZE = 64 #32 #10
LEARNING_RATE = 1.0e-5 #1.0e-5
TRAJECTORY_LEN = 5 #10
EPOCHS = 1 #10
LOG_FREQUENCY = 100
STATE_DIM = 227 #talos: 57 #quad aliengo: 227
ACT_DIM = 12 #talos: 22 #quad aliengo: 12
ALPHA = 0.1

loss_fcn = nn.MSELoss()
replay_buffer = memory.ReplayMemory(1000)
running_loss = torch.tensor([], dtype=dtype, device=device)

trajectory_seq =  []
state_seq = np.array([])
action_seq = np.array([])
reward_seq = np.array([])





replay_memory = memory.ReplayMemory(10000)


## Model ################
class StochasticDecisionTransformer(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        act_dim = config.act_dim
        
        self.mean_head = nn.Linear(config.hidden_size, act_dim)
        self.logstd_head = nn.Linear(config.hidden_size, act_dim)
        # nn.init.constant_(self.log_std_head.bias, -1.0)  # optional initialization
        
    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        out = super().forward(*args, **kwargs)
        hidden_states = out.hidden_states
        
        last_hidden = hidden_states[-1][:, -1, :]
        
        mean = self.mean_head(last_hidden)
        logstd = torch.clamp(self.logstd_head(last_hidden), min=-20, max=2)
        
        return out.state_preds, (mean, logstd), out.return_preds




configuration = DecisionTransformerConfig(
    state_dim = STATE_DIM,
    act_dim = ACT_DIM,
    hidden_size=128,
    max_ep_len=4096,
    action_tanh=True
)
model = StochasticDecisionTransformer(configuration).to(dtype).to(device) # DecisionTransformerModel(configuration).to(dtype).to(device)



## Optimizer ################
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#########################################






# config = torchrl.modules.DecisionTransformer.default_config()
# model = torchrl.modules.DecisionTransformer(state_dim=4, action_dim=2, config=config).to(dtype).to(device)



# model = models.DecisionTransformer(
#         d_s=STATE_DIM,
#         d_a=ACT_DIM,
#         d_model=100,
#         nhead=1,
#         num_encoder_layers=1,
#         dropout=0.1
#     ).to(dtype).to(device)



# model = DecisionTransformer(
#     state_dim=STATE_DIM,
#     act_dim=ACT_DIM,
#     max_length=max_length,
#     max_ep_len=max_ep_len,
#     hidden_size=variant['embed_dim'],
#     n_layer=variant['n_layer'],
#     n_head=variant['n_head'],
#     n_inner=4*variant['embed_dim'],
#     activation_function=variant['activation_function'],
#     n_positions=1024,
#     resid_pdrop=variant['dropout'],
#     attn_pdrop=variant['dropout'],
#     )







