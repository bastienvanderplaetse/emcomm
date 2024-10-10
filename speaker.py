import torch
from torch import nn
from torch.distributions.categorical import Categorical
from weightinit import custom_init

class Speaker(nn.Module):
    def __init__(self, msg_length, vocab_size, emb_dim, emb_img, core_state_dim, transform_func) -> None:
        super().__init__()

        self.length = msg_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_img = emb_img
        self.core_state_dim = core_state_dim

        
        self.embedder = nn.Embedding(self.vocab_size+1, self.emb_dim)
        self.mlp = nn.Linear(self.emb_dim, self.emb_dim)
        self.core_state_adapter = nn.Linear(self.emb_img, self.core_state_dim)
        if transform_func == 'custom':
            custom_init(self.core_state_adapter.weight.data)
        self.lstm = nn.LSTMCell(self.length, self.core_state_dim//2)

        # Policy Q-Value Dueling Head
        self.policy_head = nn.Linear(self.core_state_dim//2, self.vocab_size)
        self.value_head = nn.Linear(self.core_state_dim//2, 1)
        ## Dueling Head
        self.value_net = nn.Linear(self.core_state_dim//2, 1)
        self.advantage_net = nn.Linear(self.core_state_dim//2, self.vocab_size)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        # Imitation EMA score
        self.ema_score = 0
        self.avg_score = 0
        self.count_score = 0

    def forward(self, x, forcing=False, action_to_follow=None):
        batch_size = x.shape[0]
        state = self.core_state_adapter(x)
        
        h_t = state[:, :self.core_state_dim//2]
        c_t = state[:, self.core_state_dim//2:]
        prev_token = torch.LongTensor([self.vocab_size] * batch_size).cuda()

        action_l = []
        policy_logits_l = []
        action_log_prob_l = []
        entropy_l = []
        q_values_l = []
        value_l = []

        for t in range(self.length):
            input_t = self.embedder(prev_token)
            h_t, c_t = self.lstm(input_t, (h_t, c_t))

            # Policy Q-Value Dueling Head
            policy_logits = self.policy_head(h_t)
            value = self.value_head(h_t)
            
            ## Dueling Head
            state_value = self.value_net(h_t)
            advantage = self.advantage_net(h_t)
            mean_advantage = torch.mean(advantage, axis=-1, keepdim=True)
            q_values = state_value + advantage - mean_advantage

            m = Categorical(logits=policy_logits)
            if forcing == True:
                action = action_to_follow[:,t]
            elif h_t.requires_grad:
                # Training mode
                # Randomly choose next token according on distribution
                action = m.sample()
            else:
                # Inference mode
                # Choose highest probable token
                action = torch.argmax(policy_logits, axis=-1)
            
            action_log_prob = m.log_prob(action)
            entropy = m.entropy()

            prev_token = action
            
            action_l.append(action)
            policy_logits_l.append(policy_logits)
            entropy_l.append(entropy)

            action_log_prob_l.append(action_log_prob)
            q_values_l.append(q_values)
            value_l.append(value)

        action_l = torch.stack(action_l, axis=-1)
        policy_logits_l = torch.stack(policy_logits_l, axis=-1)
        action_log_prob_l = torch.stack(action_log_prob_l, axis=-1)
        entropy_l = torch.stack(entropy_l, axis=-1)
        q_values_l = torch.stack(q_values_l, axis=-1)
        value_l = torch.stack(value_l, axis=-1)
        
        return action_l, policy_logits_l, action_log_prob_l, entropy_l, q_values_l, value_l