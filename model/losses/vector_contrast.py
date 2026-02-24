import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    """
    Custom loss function to minimize cosine similarity between all unique pairs of vectors.
    
    Args:
        margin (float, optional): Threshold above which similarities are penalized. 
                                  If set to 0, all similarities are penalized.
                                  Default is 0.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'.
                                   'none': No reduction will be applied.
                                   'mean': The sum of the output will be divided by the number of elements.
                                   'sum': The output will be summed.
                                   Default is 'mean'.
    """
    def __init__(self, margin=0.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Expected 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, vectors):
        """
        Forward pass to compute the loss.
        
        Args:
            vectors (torch.Tensor): Input tensor of shape [num_vectors, dim]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure input is a 2D tensor
        if vectors.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, got {vectors.dim()}D tensor instead.")

        num_vectors = vectors.size(0)
        if num_vectors < 2:
            raise ValueError("Need at least two vectors to compute cosine similarity.")

        # Normalize the vectors to unit length
        normalized_vectors = F.normalize(vectors, p=2, dim=1)  # Shape: [num_vectors, dim]

        # Compute cosine similarity matrix
        cosine_sim_matrix = torch.matmul(normalized_vectors, normalized_vectors.t())  # Shape: [num_vectors, num_vectors]

        # Create a mask to exclude self-similarity (diagonal elements)
        device = vectors.device
        mask = torch.eye(num_vectors, device=device).bool()
        cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0.0)  # Shape: [num_vectors, num_vectors]

        # Depending on the margin, compute the loss
        if self.margin > 0.0:
            # Penalize similarities above the margin
            loss_matrix = F.relu(cosine_sim_matrix - self.margin)
        else:
            # Penalize all positive similarities
            loss_matrix = cosine_sim_matrix

        # Since the similarity matrix is symmetric, we consider only the upper triangle to avoid double-counting
        # Create a mask for the upper triangle
        upper_tri_mask = torch.triu(torch.ones_like(cosine_sim_matrix), diagonal=1).bool()
        loss_matrix = loss_matrix.masked_select(upper_tri_mask)

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss_matrix.mean()
        elif self.reduction == 'sum':
            loss = loss_matrix.sum()
        else:  # 'none'
            loss = loss_matrix

        return loss