import torch
import torch.distributed as dist
from torchmetrics import Metric
from einops import pack, rearrange

class CodeBookMetric(Metric):
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.index_count = None

    def update(self, indices):
        indices, ps = pack([indices], "* g")
        indices = rearrange(indices, "n g -> g n")
        g, _ = indices.shape

        index_counts = []
        for group in range(g):
            index_counts +=[
                torch.bincount(
                    indices[group].view(-1).long(), minlength=self.num_embeddings
                )
            ]

        if self.index_count is None:
            self.index_count = index_counts
        else:
            for group in range(g):
                self.index_count[group] = self.index_count[group] + index_counts[group]

    def compute_independent_codebook(self):
        def compute_group_metrics(index_count):
            # Compute probabilities for each group
            group_probs = index_count / torch.sum(index_count)

            # Group perplexity
            group_perplexity = torch.exp(
                -torch.sum(group_probs * torch.log(group_probs + 1e-10))
            ).item()

            # Group used codebook percentage
            group_used_codebook = (
                torch.count_nonzero(group_probs).item() / group_probs.numel()
            )

            return group_probs, group_perplexity, group_used_codebook

        # Compute metrics for each group
        total_perplexity = 0
        total_used_codebook = 1
        for group in range(len(self.index_count)):
            _, group_perplexity, group_used_codebook = compute_group_metrics(self.index_count[group])
            total_perplexity += group_perplexity
            total_used_codebook *= group_used_codebook

        # Aggregate metrics across groups
        total_perplexity /= len(self.index_count)  # Average perplexity across groups
        total_used_codebook = total_used_codebook * 100  # Convert to percentage

        return total_perplexity, total_used_codebook

    def compute(self):
        # Aggregate counts across all groups
        total_index_count = torch.zeros_like(self.index_count[0])
        for group in range(len(self.index_count)):
            total_index_count += self.index_count[group]

        # Compute shared probabilities for the codebook
        shared_probs = total_index_count / torch.sum(total_index_count)

        # Compute shared perplexity
        shared_perplexity = torch.exp(
            -torch.sum(shared_probs * torch.log(shared_probs + 1e-10))
        ).item()

        # Compute shared used codebook percentage
        shared_used_codebook = (
            torch.count_nonzero(total_index_count).item() / total_index_count.numel()
        ) * 100

        return shared_perplexity, shared_used_codebook
    
    def reset(self):
        self.index_count = None

    def reduce(self):
        for group in range(len(self.index_count)):
            tensor = self.index_count[group]
            dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            self.index_count[group] = tensor
            
    def get_result(self):
        self.reduce()
        perplexity, used_codebook = self.compute()
        prod_perplexity, prod_used_codebook = self.compute_independent_codebook()

        output = {
            "perplexity": perplexity,
            "used_codebook": used_codebook,
            "prod_perplexity": prod_perplexity,
            "prod_used_codebook": prod_used_codebook,
        }

        return output