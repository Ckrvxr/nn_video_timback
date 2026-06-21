import torch

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) Optimizer Wrapper.
    
    SAM seeks parameters that lie in neighborhoods of uniformly low loss,
    leading to flatter minima and better generalization.
    
    Usage:
        optimizer = SAM(model.parameters(), base_optimizer=torch.optim.AdamW, rho=0.05, lr=1e-3, ...)
        
        # First pass
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Second pass
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize base optimizer using the same parameters and arguments
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        # Check if grad_norm is invalid (zero, NaN, or Inf)
        is_invalid = torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item() or grad_norm.item() == 0
        
        for group in self.param_groups:
            if is_invalid:
                scale = 0.0
            else:
                scale = group["rho"] / (grad_norm + 1e-8)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                # Save the original parameters
                self.state[p]["old_p"] = p.data.clone()
                if scale > 0.0:
                    # Calculate adversarial perturbation
                    e_w = (p.grad * scale).to(p)
                    # Climb to the local maximum "adversarial" parameter
                    p.add_(e_w)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    # Restore the original parameters
                    p.data.copy_(self.state[p]["old_p"])
                    # Delete the copy to free memory
                    del self.state[p]["old_p"]

        # Step the base optimizer using the accumulated gradients at the adversarial point
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires a two-step optimization loop: first_step() and second_step(). "
            "Please use the custom first_step/second_step loop in training."
        )

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups 
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
