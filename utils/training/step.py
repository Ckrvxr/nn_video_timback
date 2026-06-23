from typing import Callable

import torch
import torch.nn as nn

from utils.console import console


def has_nan_or_inf_gradients(model: torch.nn.Module) -> bool:
    """Check if any model gradients contain NaN or Inf values."""
    return any(
        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
        for p in model.parameters()
        if p.grad is not None
    )


def sam_first_pass(model, optimizer, scaler, config, run_forward: Callable):
    moe_weight = config['loss_weights'].get('moe', 0.01)

    if scaler is not None:
        scaler.unscale_(optimizer)

    if has_nan_or_inf_gradients(model):
        console.warning("NaN or Inf detected in gradients. Skipping SAM step.")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update()
        return None
    return moe_weight


def sam_second_pass(model, optimizer, scaler, clip_grad, run_forward: Callable, saved_state):
    optimizer.first_step(zero_grad=True)

    if saved_state is not None:
        model._t_state = saved_state.clone()

    with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
        loss_adv, loss_dict_adv = run_forward()

    if scaler is not None:
        scaler.scale(loss_adv).backward()
        inv_scale = 1.0 / (scaler.get_scale() + 1e-8)
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(inv_scale)
    else:
        loss_adv.backward()

    if has_nan_or_inf_gradients(model):
        console.warning("NaN or Inf detected in second-pass gradients. Skipping SAM step.")
        if scaler is not None:
            opt_state = scaler._per_optimizer_states.get(id(optimizer))
            if opt_state is not None:
                for dev in opt_state['found_inf_per_device'].keys():
                    opt_state['found_inf_per_device'][dev].fill_(1.0)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state and "old_p" in optimizer.state[p]:
                    p.data.copy_(optimizer.state[p]["old_p"])
                    del optimizer.state[p]["old_p"]
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update()
    else:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.second_step(zero_grad=True)
        if scaler is not None:
            scaler.update()


def standard_optimizer_step(model, optimizer, scaler, clip_grad):
    if scaler is not None:
        scaler.unscale_(optimizer)
        if has_nan_or_inf_gradients(model):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
    else:
        if has_nan_or_inf_gradients(model):
            optimizer.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def log_expert_utilization(expert_counts, loss_dict, batch_loss, optimizer, epoch, n_epochs, batch_idx, n_batches):
    total_selections = expert_counts.sum().item()
    if total_selections > 0:
        n_experts = expert_counts.size(0)
        console.opt(colors=True).info(
            "<bold><cyan>═══ Expert Utilization (Epoch {}/{}, Batch {}/{}) ═══</cyan></bold>",
            epoch + 1, n_epochs, batch_idx + 1, n_batches,
        )
        console.opt(colors=True).info("<dim>Legend: </dim>"
            "<red>≥20%</red> <dim>|</dim> <yellow>≥10%</yellow> <dim>|</dim> "
            "<green>≥5%</green> <dim>|</dim> <blue>≥1%</blue> <dim>|</dim> <dim>&lt;1%</dim>")

        def heat_color(pct):
            if pct >= 20: return "red"
            if pct >= 10: return "yellow"
            if pct >= 5:  return "green"
            if pct >= 1:  return "blue"
            return "dim"

        from utils.console import divider
        for row_start in range(0, n_experts, 10):
            cells = []
            for offset in range(10):
                idx = row_start + offset
                if idx >= n_experts:
                    break
                cnt = expert_counts[idx].item()
                pct = (cnt / total_selections) * 100
                color = heat_color(pct)
                cells.append(f"<{color}>E{idx:02d} {pct:>5.1f}%</{color}>")
            console.opt(colors=True).info("  ".join(cells))
        divider()
        top5 = expert_counts.topk(5)
        top_pairs = "  ".join(
            f"E{idx} {cnt/total_selections*100:.1f}%"
            for idx, cnt in zip(top5.indices.tolist(), top5.values.tolist())
        )
        loss_parts = "  ".join(
            f"{name}={val.item():.6f}"
            for name, val in loss_dict.items() if name != 'total'
        )
        lr_val = optimizer.param_groups[0]['lr']
        console.opt(colors=True).info(
            '<level>{}</level>  <green>loss</green>=<yellow>{:.6f}</yellow>  <cyan>lr={:.2e}</cyan>',
            loss_parts, batch_loss, lr_val,
        )
        console.opt(colors=True).info('<green>Top-5:</green> {}', top_pairs)
        divider()
