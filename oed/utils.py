import torch

def initialize_candidate_designs(num_candidates=200, seed=777):
    ranges = [torch.arange(-4, 4.1, 0.1) for _ in range(2)]
    grid = torch.cartesian_prod(*ranges)
    torch.manual_seed(seed)
    indices = torch.randperm(len(grid))[:num_candidates]
    candidate_designs = grid[indices]
    return candidate_designs


def compute_mmd_dispersion_batch(points_batch, uniform_samples=None, sigma=1.0):
    """
    Batch version of compute_mmd_dispersion.
    
    Parameters:
    - points_batch (Tensor): shape (batch_size, 30, 2)
    - uniform_samples (Tensor): shape (num_candidates, 2)
    - sigma (float): RBF kernel bandwidth

    Returns:
    - mmds (Tensor): shape (batch_size,)
    """
    batch_size = points_batch.shape[0]
    if uniform_samples is None:
        uniform_samples = initialize_candidate_designs().to(points_batch.device)  # (num_candidates, 2)

    # Expand uniform_samples to batch dimension
    uniform_samples = uniform_samples.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_candidates, 2)

    K_xx = rbf_kernel_batch(points_batch, points_batch, sigma)  # (batch_size,)
    K_yy = rbf_kernel_batch(uniform_samples, uniform_samples, sigma)  # (batch_size,)
    K_xy = rbf_kernel_batch(points_batch, uniform_samples, sigma)  # (batch_size,)

    m = points_batch.shape[1]
    n = uniform_samples.shape[1]

    mmds = (K_xx / (m * m)) + (K_yy / (n * n)) - (2 * K_xy / (m * n))
    return mmds

def rbf_kernel_batch(x, y, sigma=1.0):
    """
    Computes RBF kernel between two batches of samples.
    
    x: (batch_size, m, d)
    y: (batch_size, n, d)
    
    Returns:
    - kernel_sum: (batch_size,)  # Sum over all pairs
    """
    # x -> (batch_size, m, 1, d)
    # y -> (batch_size, 1, n, d)
    x = x.unsqueeze(2)  # (batch_size, m, 1, d)
    y = y.unsqueeze(1)  # (batch_size, 1, n, d)
    diff = x - y  # (batch_size, m, n, d)
    dist_sq = (diff ** 2).sum(-1)  # (batch_size, m, n)

    K = torch.exp(-dist_sq / (2 * sigma ** 2))  # (batch_size, m, n)

    return K.sum(dim=(1,2))  # Sum over (m,n) for each batch