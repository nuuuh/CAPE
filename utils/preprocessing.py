import torch
import torch.nn as nn
import numpy as np
import random

from .revin import RevIN

class preprocess(nn.Module):
    def __init__(self, num_features=None, normalization=True, patching=False, norm_affine=False, masking=False, patch_len=12, stride=1):
        super(preprocess, self).__init__()
        self.norm = normalization
        self.patching = patching
        self.patch_len = patch_len
        self.stride = stride
        self.masking = masking
        if self.norm:
            self.revin = RevIN(num_features, affine=norm_affine)
    
    def forward(self, x, y=None, time=None,  mode='norm', mask_prob=0.0, prob=0.1, peak_mask=0.0):
        if self.norm:
            x = self.revin(x.float(), mode)
            if y is not None:
                y = self.revin._normalize(y)
        if self.patching:
            # import ipdb; ipdb.set_trace()
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            if time is not None:
                time = time.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if self.masking:
            weight = None
            # import ipdb; ipdb.set_trace()
            select_prob = random.uniform(0,1)
            if select_prob < mask_prob:
                # random masking (RANDMASK)
                masking = self.random_masking(x, prob=prob)
                weight = 0.6
            elif select_prob < mask_prob + peak_mask:
                # peak masking (PEAKMASK)
                masking = self.peak_masking(x)
                weight = 0.5
            else:
                # masking last value (default)
                masking = self.last_mask(x, num=1)
                weight = 0.4

                # randomly injecting noise

            masked_x = x * masking
            labels = x * ~masking
            if time is not None:
                return masked_x, labels, masking, weight, time
            else:
                return masked_x, labels, masking, weight
            
        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        if time is not None:
            return x, y, time
        else:
            return x, y

    def denorm(self, x):
        return self.revin(x, mode='denorm')

    def norm_single_value(self, x):
        return self.revin._normalize(x)

    def random_noise_masking(self, ts, ts_length):
        ts_masking = ts.copy()
        mask = np.zeros((self.seq_len,), dtype=int)
        for i in range(ts_length):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                mask[i] = 1

                if prob < 0.5:
                    ts_masking[i, :] += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))

                else:
                    ts_masking[i, :] += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

        return ts_masking, mask
    
    def random_masking(self, x, prob):
        mask = torch.zeros_like(x)
        prob_mask = torch.rand(x.shape[0],x.shape[1])

        mask[prob_mask >= prob] = 1
        mask = mask.bool()
        return mask


    def last_mask(self, x, num=1):
        mask = torch.ones_like(x)
        mask = mask.bool()
        # import ipdb; ipdb.set_trace()
        mask[:,-num:,:] = False

        return mask

    def peak_masking(self, x):
        """
        Peak masking: Mask all segments that cover the timestamp with maximum value.
        For each sample in the batch, identify the time stamp with the maximum value.
        Then mask all patches/segments that cover this maximum value's time-stamp.
        
        Args:
            x: Input tensor of shape [batch, seq_len, features] or [batch, num_patches, patch_len]
        
        Returns:
            mask: Boolean mask tensor (True = keep, False = mask)
        """
        mask = torch.ones_like(x).bool()
        
        # Handle patched data: [batch, num_patches, patch_len]
        if len(x.shape) == 3 and self.patching:
            batch_size, num_patches, patch_len = x.shape
            
            # Reconstruct the original sequence to find the peak position
            # We need to map patches back to time indices
            for b in range(batch_size):
                # Find max value across all patches and patch positions
                max_val = x[b].max()
                
                # Find which patch(es) contain the maximum value
                for p in range(num_patches):
                    if (x[b, p] == max_val).any():
                        # Mask this entire patch
                        mask[b, p, :] = False
        else:
            # Handle non-patched data: [batch, seq_len, features]
            batch_size = x.shape[0]
            for b in range(batch_size):
                # Find the index of maximum value in the sequence
                max_idx = x[b].max(dim=-1)[0].argmax()
                # Mask the time step with the maximum value
                mask[b, max_idx, :] = False
        
        return mask


def take_per_row(A, indx, num_elem):
    """
        Takes num_elements starting at indx for each batch element.
    """
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]
