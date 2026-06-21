import math

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_peak, true_peak, min_lr, warmup_epochs, n_epochs):
        self.optimizer = optimizer
        self.warmup_peak = warmup_peak
        self.true_peak = true_peak
        self.min_lr = min_lr
        self.warmup_epochs = max(warmup_epochs, 0)
        self.n_epochs = max(n_epochs, 1)
        self.last_epoch = 0
        self.base_lrs = [warmup_peak]
        self._step_count = 0

        init_lr = warmup_peak if self.warmup_epochs > 0 else true_peak
        self._last_lr = [init_lr]
        for pg in optimizer.param_groups:
            pg['lr'] = init_lr

    def _compute_lr(self, epoch):
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            t = epoch / self.warmup_epochs
            factor = math.cos(math.pi * 0.5 * t)
            return self.true_peak + (self.warmup_peak - self.true_peak) * factor

        effective_n = max(self.n_epochs - self.warmup_epochs, 1)
        t = (epoch - self.warmup_epochs) / effective_n
        factor = (1 + math.cos(math.pi * t)) / 2
        return self.min_lr + (self.true_peak - self.min_lr) * factor

    def step(self):
        self._step_count += 1
        self.last_epoch += 1
        lr = self._compute_lr(self.last_epoch)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self._last_lr = [lr]

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            '_step_count': self._step_count,
            '_last_lr': self._last_lr,
            'warmup_peak': self.warmup_peak,
            'true_peak': self.true_peak,
            'min_lr': self.min_lr,
            'warmup_epochs': self.warmup_epochs,
            'n_epochs': self.n_epochs,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self._step_count = state_dict['_step_count']
        self._last_lr = state_dict['_last_lr']
        self.warmup_peak = state_dict['warmup_peak']
        self.true_peak = state_dict['true_peak']
        self.min_lr = state_dict['min_lr']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.n_epochs = state_dict['n_epochs']
        for pg, lr in zip(self.optimizer.param_groups, self._last_lr):
            pg['lr'] = lr
