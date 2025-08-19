import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, DecisionTransformerModel, OpenAIGPTConfig, OpenAIGPTModel, AutoConfig
import matplotlib; matplotlib.use("Agg")
from collections import namedtuple, deque





## device & dtype & direction ################################
device = torch.device('mps')
dtype = torch.float32
result_dir = "/Users/user/Documents/Professional/University of Manchester/Academic/Research/10. Trajectory Optimization - Policy Search/Code/results/"
##################################################


## global parameters #############################
loss_fcn = nn.MSELoss()
window_size = 1
loss_save = torch.tensor([]).to(dtype).to(device)
trajectory_length = 1000
##################################################



## model
# transformer_model_name = "sshleifer/tiny-gpt2" #"gpt2" # "edbeeching/decision-transformer-gym-halfcheetah-expert"
# tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
# transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_name)
# config = AutoConfig.from_pretrained(transformer_model_name)
# transformer_model = AutoModelForCausalLM.from_config(config)

# transformer_model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)




class GPT2Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        return x

class GPT2FloatInputModel(nn.Module):
    def __init__(self, d_model=100, nhead=1, num_layers=1,
                 dim_feedforward=4*100, dropout=0.1, max_seq_len=trajectory_length, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList([
            GPT2Block(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: (B, T)
        B, T = x.shape
        assert T == self.pos_emb.size(1), f"Expected input length {self.pos_emb.size(1)}, got {T}"

        x = x.unsqueeze(-1)            # (B, T, 1)
        x = self.input_proj(x)         # (B, T, d_model)
        x = x + self.pos_emb[:, :T, :] # Add positional embedding

        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self.head(x)


transformer_model = GPT2FloatInputModel()
transformer_model.requires_grad_().train().to(dtype).to(device)
print(transformer_model)
print("Num. of Parameters = ", sum(p.numel() for p in transformer_model.parameters()))

# for j, (name, param) in enumerate(transformer_model.named_parameters()):
#     if 'weight' in name:
#         torch.nn.init.kaiming_normal_(param)

peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.01,
            target_modules = ["attn"],
            bias="none",
            modules_to_save=["classifier"]
            )

model = transformer_model #!!!!!!!
# model = get_peft_model(transformer_model, peft_config)
# model.print_trainable_parameters()



## Optimizer ################
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
#########################################


## Class #########################################

Trajectory = namedtuple('Trajectory', ('trajectory_mpc', 'action_mpc'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, trajectory_mpc, action_mpc):
        """Save a trajectory"""
        self.memory.append(Trajectory(trajectory_mpc, action_mpc))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)





## Functions #####################################

def flatten_state(state):
    return np.concatenate([np.ravel(v) for v in state.values()])


def min_max(min_vals, max_vals):
    global min_traj, max_traj, D
    min_traj = min_vals
    max_traj = max_vals
    D = min_traj.shape[0]
    return True

"""
def uniform_discretize(X, V=100):
    batch_size, total_D = X.shape
    i = 0
    tokens = torch.zeros_like(X, dtype=torch.long)
    for d in range(total_D):
        if d % D == 0:
            i = 0
        bins = torch.linspace(min_traj[i], max_traj[i], V + 1, device=X.device, dtype=X.dtype)
        token_ids = torch.bucketize(X[:, d], bins) - 1
        token_ids = torch.clamp(token_ids, 0, V - 1)
        tokens[:, d] = token_ids + i * V
        i+=1
    return tokens


def uniform_discretize_inverse(output_token, V=100):
    batch_size, action_size = output_token.shape
    output_predict = torch.zeros_like(output_token, dtype=torch.float64)
    for d in range(action_size):
        # bins = torch.linspace(min_traj[-action_size+d], max_traj[-action_size+d], V + 1, device=output_token.device, dtype=torch.float32)
        output_predict[:,d] = min_traj[-action_size+d] + (output_token[:,d] - (D-action_size+d)*V) * (max_traj[-action_size+d]-min_traj[-action_size+d])/V
    return output_predict


def quantile_discretize(X, V=100):
    ## incorrect implementation!!!!!!
    batch_size, total_D = X.shape
    i = 0
    tokens = torch.zeros_like(X, dtype=torch.long)
    for d in range(total_D):
        quantiles = torch.quantile(X[:, d], q=torch.linspace(0, 1, V + 1, device=X.device, dtype=X.dtype))
        token_ids = torch.bucketize(X[:, d], quantiles[1:-1]) - 1
        token_ids = torch.clamp(token_ids, 0, V - 1)
        tokens[:, d] = token_ids + i * V
        i+=1
        if d % D == 0:
            i = 0
    return tokens
"""


def my_discretize(X, V=100):
    X = X * V + 30000
    X = torch.round(X).int()
    X = torch.clamp(X,min=0,max=50000)
    return X


def my_discretize_inverse(X, V=100):
    X = (X - 30000).float() / float(V)
    return X







def policy(trajectory, action):

    global loss_save

    action = torch.from_numpy(action).requires_grad_().to(dtype).to(device)
    trajectory = torch.from_numpy(trajectory).requires_grad_().to(dtype).to(device)
    trajectory_window = trajectory_length #3*D
    action_size = 12
    batch_size = trajectory.shape[0]

    # input = my_discretize(trajectory)
    input = trajectory    

    
    output = [] # torch.zeros((batch_size, action_size)).to(dtype).to(device)
    for i in range(action_size):
        # probs = torch.softmax(model(input).logits[:,-1,:], dim=-1)
        
        logits = model(input)
        # probs = torch.softmax(logits[:, -1, :], dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)
        # next_token = F.gumbel_softmax(logits[:, -1, :], tau=0.1, hard=True)
        # output[:,i].add_(next_token.squeeze(1))
        next_token = logits[:,-1,:]
        output.append(next_token)
        input = torch.cat([input, next_token], dim=1)
        input = input[:, -trajectory_window:]
        
    output_predict = torch.stack(output, dim=1)
    output_predict = output_predict.reshape(action.shape)
    # output_predict = my_discretize_inverse(output, V=100)
    # output_predict = output

    

    # print("input = ", input.shape)
    # print("output = ", output.shape)
    # print("Predicted action = ", output_predict[0])
    # print("Real action = ", action[0])

    loss = loss_fcn(output_predict, action)
    loss.backward()

    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         pass
    #     else:
    #         print(f"{name} has gradient with norm {param.grad.norm().item():.4f}")

    optimizer.step()
    optimizer.zero_grad()


    ## save & print #####################################
    print("-------------------------")
    # print("Loss = ", loss.item())
    print("Predicted action = ", output_predict[0])
    print("Real action = ", action[0])

    loss_save = torch.cat([loss_save, loss.detach().unsqueeze(0)])
    torch.save(loss_save, result_dir + 'loss_adam.pt')
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(np.convolve(np.array(loss_save.cpu()), np.ones(window_size)/window_size, mode='valid'), label='Adam', color='darkred')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig(result_dir+'loss_adam.pdf')
    plt.close()
    
    if len(loss_save) >= 2:
        del_loss = loss_save[-2] - loss_save[-1]
    else:
        del_loss = 100.0

    return del_loss
