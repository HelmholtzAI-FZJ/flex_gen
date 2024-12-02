import torch
import torch.distributed as dist
from torchmetrics import Metric


class CodeBookMetric(Metric):
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.index_count = None

    def update(self, indices):
        index_count = torch.bincount(
            indices.view(-1).long(), minlength=self.num_embeddings
        )
        if self.index_count is None:
            self.index_count = index_count
        else:
            self.index_count = self.index_count + index_count

    def compute(self):
        index_count = self.index_count
        # get used idx as probabilities
        used_indices = index_count / torch.sum(index_count)

        # perplexity
        perplexity = torch.exp(
            -torch.sum(used_indices * torch.log(used_indices + 1e-10), dim=-1)
        ).item()

        # get the percentage of used codebook
        n = index_count.shape[0]
        used_codebook = (torch.count_nonzero(used_indices).item() * 100) / n

        return used_indices, perplexity, used_codebook

    def reset(self):
        self.index_count = None

    def reduce(self):
        tensor = self.index_count
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        self.index_count = tensor

    def get_result(self):
        self.reduce()
        used_indices, perplexity, used_codebook = self.compute()

        output = {
            "perplexity": perplexity,
            "used_codebook": used_codebook,
        }

        return output
