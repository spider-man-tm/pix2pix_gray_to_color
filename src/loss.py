import torch


class HingeLoss():
    def __init__(self, condition):
        self.condition = condition     

    def __call__(self, logits, patch):
        assert self.condition in ['gen', 'dis_real', 'dis_fake']
        batch_len = len(logits)
        if self.condition == 'gen':
            return -torch.mean(logits)
        elif self.condition == 'dis_real':
            minval = torch.min(logits - 1, patch)
            return -torch.mean(minval)
        else:
            minval = torch.min(-logits - 1, patch)
            return - torch.mean(minval)