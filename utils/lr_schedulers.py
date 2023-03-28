from abc import ABCMeta, abstractmethod
import math
import torch

class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass

class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)
            
class CosineAnnealingLR(BaseLR):
    def __init__(self, start_lr, min_lr, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        return self.min_lr + 0.5 * (self.start_lr - self.min_lr) * \
            (1+math.cos(math.pi * cur_iter / (self.total_iters - self.warmup_steps)))

    