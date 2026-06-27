import torch


class SophiaG(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), rho=0.04,
                 weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            rho = group['rho']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['h'] = torch.zeros_like(p)

                m, h = state['m'], state['h']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                ratio = m / (h + 1e-8)
                ratio.clamp_(-rho, rho)
                p.data.add_(ratio, alpha=-lr)

        return loss

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'h' not in state:
                    state['h'] = torch.zeros_like(p)
                h = state['h']
                h.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)
