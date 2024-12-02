import torch
import torch.distributed as dist

class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    # def update(self, batch):
    #     """
    #     Update statistics with a new batch of data.
    #     :param batch: Tensor of shape (batch_size, features)
    #     """
    #     batch = batch.detach()
    #     batch = batch.view(batch.size(0), -1)  # Flatten if necessary
    #     batch_size, features = batch.size()
        
    #     # Compute batch statistics
    #     batch_mean = batch.mean(dim=0)
    #     batch_var = batch.var(dim=0, unbiased=False)
    #     batch_M2 = batch_var * batch_size

    #     # Update global statistics
    #     delta = batch_mean - self.mean
    #     total_n = self.n + batch_size

    #     new_mean = self.mean + delta * batch_size / total_n
    #     new_M2 = self.M2 + batch_M2 + delta.pow(2) * self.n * batch_size / total_n

    #     self.mean = new_mean
    #     self.M2 = new_M2
    #     self.n = total_n

    def update(self, batch):
        batch = batch.detach()
        batch = batch.view(-1)  # Flatten all elements
        batch_size = batch.numel()
        batch_mean = batch.mean()
        batch_var = batch.var(unbiased=False)
        batch_M2 = batch_var * batch_size

        delta = batch_mean - self.mean
        total_n = self.n + batch_size

        new_mean = self.mean + delta * batch_size / total_n
        new_M2 = self.M2 + batch_M2 + (delta ** 2) * self.n * batch_size / total_n

        self.mean = new_mean
        self.M2 = new_M2
        self.n = total_n

    def get_stats(self):
        return self.n, self.mean, self.M2

    @staticmethod
    def merge_stats(n1, mean1, M21, n2, mean2, M22):
        delta = mean2 - mean1
        total_n = n1 + n2
        if total_n == 0:
            return 0, torch.zeros_like(mean1), torch.zeros_like(M21)
        new_mean = mean1 + delta * n2 / total_n
        new_M2 = M21 + M22 + delta.pow(2) * n1 * n2 / total_n
        return total_n, new_mean, new_M2


def compute_global_stats(local_stats):
    """
    Compute global mean and variance from local statistics.
    :param local_stats: Tuple (n, mean, M2) from local rank
    :return: Tuple (global_mean, global_variance)
    """
    n, mean, M2 = local_stats

    # Prepare tensors for all_reduce
    device = mean.device
    n_tensor = torch.tensor([n], dtype=torch.float64, device=device)
    mean_tensor = mean.clone()
    M2_tensor = M2.clone()

    # All-reduce the statistics
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(M2_tensor, op=dist.ReduceOp.SUM)

    total_n = n_tensor.item()
    if total_n == 0:
        global_mean = torch.zeros_like(mean)
        global_var = torch.zeros_like(M2)
    else:
        global_mean = mean_tensor / total_n
        global_var = M2_tensor / total_n

    return global_mean, global_var
