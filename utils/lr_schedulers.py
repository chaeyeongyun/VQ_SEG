from abc import ABCMeta, abstractmethod
import math
import torch
import logging

def get_scheduler(lr_args, len_data, num_epochs, optimizer, start_epoch=0):
    lr_scheduler = LRScheduler(
        lr_args, len_data, optimizer, num_epochs, start_epoch
    )
    return lr_scheduler

class LRScheduler(object):
    def __init__(self, lr_args, data_size, optimizer, num_epochs, start_epochs):
        super(LRScheduler, self).__init__()
        logger = logging.getLogger("global")
        name = lr_args['name']
        assert name in ["multistep", "poly", "cosineannealing"]
        self.name = name
        self.optimizer = optimizer
        self.data_size = data_size

        self.cur_iter = start_epochs * data_size
        self.max_iter = num_epochs * data_size

        # set learning rate
        self.base_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.cur_lr = [lr for lr in self.base_lr]
        
        if name == "poly":
            self.power = lr_args["power"] if lr_args.get("power", False) else 0.9
            logger.info("The kwargs for lr scheduler: {}".format(self.power))
        if name == "milestones":
            default_mist = list(range(0, num_epochs, num_epochs // 3))[1:]
            self.milestones = (
                lr_args["milestones"]
                if lr_args.get("milestones", False)
                else default_mist
            )
            logger.info("The kwargs for lr scheduler: {}".format(self.milestones))
        if name == "cosineannealing":
            self.targetlr = lr_args["min_lr"]
            logger.info("The kwargs for lr scheduler: {}".format(self.targetlr))

    def step(self):
        self._step()
        self.update_lr()
        self.cur_iter += 1

    def _step(self):
        if self.name == "multistep":
            epoch = self.cur_iter // self.data_size
            power = sum([1 for s in self.milestones if s <= epoch])
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * pow(0.1, power)
                self.cur_lr[i] = adj_lr
        elif self.name == "poly":
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * (
                    (1 - float(self.cur_iter) / self.max_iter) ** (self.power)
                )
                self.cur_lr[i] = adj_lr
        elif self.name == "cosineannealing":
            for i, lr in enumerate(self.base_lr):
                adj_lr = (
                    self.targetlr
                    + (lr - self.targetlr)
                    * (1 + math.cos(math.pi * self.cur_iter / self.max_iter))
                    / 2
                )
                self.cur_lr[i] = adj_lr
        else:
            raise NotImplementedError

    def get_lr(self):
        return self.cur_lr

    def update_lr(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.cur_lr):
            param_group["lr"] = lr

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
