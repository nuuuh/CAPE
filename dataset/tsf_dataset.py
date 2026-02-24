from torch.utils.data import Dataset
import torch
import numpy as np

class FinetuneDataset(Dataset):
    def __init__(self, data, feature_num, seq_len, scalar=None):
        self.seq_len = seq_len
        self.dimension = feature_num
        self.data = data['data']
        self.time = data['time']
        self.scalar = scalar
        self.DFE_value = self.scalar.transform(torch.tensor([0]).reshape(1,-1)).item()
        # import ipdb; ipdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        time = self.time[idx]

        history, future = sample[0], sample[1]
        history_time, future_time = time[0], time[1]
        # import ipdb; ipdb.set_trace()

        return {'input': torch.FloatTensor(history), 'label': torch.FloatTensor(future), 'input_time': torch.FloatTensor(history_time)%100, 'output_time': torch.FloatTensor(future_time)%100, "idx": idx}


class PretrainDataset(Dataset):
    def __init__(self, data, feature_num, seq_len):
        self.seq_len = seq_len
        self.dimension = feature_num
        self.data = data['data']
        self.time = data['time']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        time = self.time[idx]

        history, _ = sample[0], sample[1]
        time, _ = time[0], time[1]

        # import ipdb; ipdb.set_trace()

        return {'input': torch.FloatTensor(history), 'input_time': torch.FloatTensor(time)%100}


class NextTokenPretrainDataset(Dataset):
    """Dataset for next-token-prediction pretraining (autoregressive)"""
    def __init__(self, data, feature_num, seq_len, token_size=4):
        self.seq_len = seq_len
        self.dimension = feature_num
        self.token_size = token_size
        self.data = data['data']
        self.time = data['time']
        
        # Filter samples that can be tokenized
        self.valid_indices = []
        for idx in range(len(self.data)):
            history = self.data[idx][0]
            if len(history) >= token_size * 2:  # Need at least 2 tokens
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.data[real_idx]
        time = self.time[real_idx]

        history, _ = sample[0], sample[1]
        time_seq, _ = time[0], time[1]
        
        # Reshape into tokens: [seq_len] -> [num_tokens, token_size]
        seq_len = len(history)
        num_tokens = seq_len // self.token_size
        
        # Truncate to fit token_size
        truncated_len = num_tokens * self.token_size
        history = history[:truncated_len]
        time_seq = time_seq[:truncated_len]
        
        # Reshape to tokens
        tokens = history.reshape(num_tokens, self.token_size)
        time_tokens = time_seq.reshape(num_tokens, self.token_size)
        
        # For pretraining, we create input-target pairs for autoregressive prediction
        # Input: tokens[:-1], Target: tokens[1:]
        if num_tokens > 1:
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            input_time = time_tokens[:-1]
        else:
            # Fallback if only 1 token (shouldn't happen with filtering)
            input_tokens = tokens
            target_tokens = tokens
            input_time = time_tokens
        
        return {
            'input': torch.FloatTensor(input_tokens),  # [num_tokens-1, token_size]
            'target': torch.FloatTensor(target_tokens),  # [num_tokens-1, token_size]
            'input_time': torch.FloatTensor(input_time) % 100  # [num_tokens-1, token_size]
        }


class NextTokenFinetuneDataset(Dataset):
    """Dataset for next-token-prediction finetuning"""
    def __init__(self, data, feature_num, seq_len, token_size=4, scalar=None):
        self.seq_len = seq_len
        self.dimension = feature_num
        self.token_size = token_size
        self.data = data['data']
        self.time = data['time']
        self.scalar = scalar
        
        # Filter valid samples
        self.valid_indices = []
        for idx in range(len(self.data)):
            history = self.data[idx][0]
            future = self.data[idx][1]
            total_len = len(history) + len(future)
            if total_len >= token_size * 2:  # Need at least 2 tokens
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.data[real_idx]
        time = self.time[real_idx]

        history, future = sample[0], sample[1]
        history_time, future_time = time[0], time[1]
        
        # Concatenate history and future for tokenization
        full_seq = np.concatenate([history, future])
        full_time = np.concatenate([history_time, future_time])
        
        # Reshape into tokens
        seq_len = len(full_seq)
        num_tokens = seq_len // self.token_size
        
        # Truncate to fit token_size
        truncated_len = num_tokens * self.token_size
        full_seq = full_seq[:truncated_len]
        full_time = full_time[:truncated_len]
        
        # Reshape to tokens
        tokens = full_seq.reshape(num_tokens, self.token_size)
        time_tokens = full_time.reshape(num_tokens, self.token_size)
        
        # Split into input (history tokens) and label (future tokens)
        num_history_tokens = len(history) // self.token_size
        
        input_tokens = tokens[:num_history_tokens]
        label_tokens = tokens[num_history_tokens:]
        input_time = time_tokens[:num_history_tokens]
        output_time = time_tokens[num_history_tokens:]
        
        return {
            'input': torch.FloatTensor(input_tokens),  # [num_history_tokens, token_size]
            'label': torch.FloatTensor(label_tokens),  # [num_future_tokens, token_size]
            'input_time': torch.FloatTensor(input_time) % 100,
            'output_time': torch.FloatTensor(output_time) % 100,
            "idx": real_idx
        }




