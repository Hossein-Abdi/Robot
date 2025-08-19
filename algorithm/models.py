import torch
import math
import torch.nn as nn



# --- 1. Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformer models.
    This adds a sinusoidal positional signal to the embeddings,
    allowing the model to understand the order of elements in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (sequence_length, batch_size, embedding_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- 2. Decision Transformer Model ---
class DecisionTransformer(nn.Module):
    """
    A simplified Decision Transformer model.
    Input: Sequence of (reward, state, action) tokens.
    Output: Predicted action for the last timestep.

    The model concatenates reward, state, and action embeddings,
    applies positional encoding, feeds them through a transformer encoder,
    and then predicts the action from the last action token's output.
    """
    def __init__(self,
                 d_s: int,         # Dimension of state
                 d_a: int,         # Dimension of action
                 d_model: int,     # Embedding dimension for transformer
                 nhead: int,       # Number of attention heads
                 num_encoder_layers: int, # Number of transformer encoder layers
                 dropout: float = 0.1):
        super().__init__()

        self.d_s = d_s
        self.d_a = d_a
        self.d_model = d_model

        # Embedders for reward, state, and action
        # Reward is 1D, so we embed it to d_model
        self.reward_embedding = nn.Linear(1, d_model)
        self.state_embedding = nn.Linear(d_s, d_model)
        self.action_embedding = nn.Linear(d_a, d_model)

        self.positional_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer to predict action
        self.action_head = nn.Linear(d_model, d_a)

    def forward(self, rewards: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: Tensor, shape (batch_size, sequence_length, 1)
            states: Tensor, shape (batch_size, sequence_length, d_s)
            actions: Tensor, shape (batch_size, sequence_length, d_a)

        Returns:
            predicted_actions: Tensor, shape (batch_size, d_a)
        """
        batch_size, seq_len, _ = rewards.shape

        # Embed rewards, states, actions
        # Output shape: (batch_size, seq_len, d_model)
        embedded_rewards = self.reward_embedding(rewards)
        embedded_states = self.state_embedding(states)
        embedded_actions = self.action_embedding(actions)

        # Interleave the embedded tokens: R1, S1, A1, R2, S2, A2, ...
        # This creates a sequence where each (R, S, A) tuple is represented by 3 tokens.
        # Total sequence length will be seq_len * 3
        # Shape: (batch_size, seq_len * 3, d_model)
        # We need to reshape for the transformer which expects (sequence_length, batch_size, d_model)
        # for batch_first=False
        interleaved_tokens = torch.empty(batch_size, seq_len * 3, self.d_model, device=rewards.device)
        interleaved_tokens[:, 0::3] = embedded_rewards
        interleaved_tokens[:, 1::3] = embedded_states
        interleaved_tokens[:, 2::3] = embedded_actions

        # Apply positional encoding (transpose to (seq_len, batch_size, d_model))
        interleaved_tokens = interleaved_tokens.permute(1, 0, 2)
        interleaved_tokens = self.positional_encoder(interleaved_tokens)

        # Pass through transformer encoder
        # Output shape: (seq_len * 3, batch_size, d_model)
        transformer_output = self.transformer_encoder(interleaved_tokens)

        # To predict action_n, we take the output corresponding to the *last action token* (A_n)
        # The last action token is at index (seq_len * 3 - 1) in the interleaved sequence.
        # We need to extract this token's output for each item in the batch.
        # Shape of last_action_token_output: (batch_size, d_model)
        last_action_token_output = transformer_output[seq_len * 3 - 1, :, :]

        # Predict the action
        predicted_action = self.action_head(last_action_token_output)

        return predicted_action
