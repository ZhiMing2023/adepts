import torch
from torch.optim.optimizer import Optimizer
from functools import reduce
import math
from ..log import LogManager

logger = LogManager()

# Adapted from torch.optim.lbfgs (BSD-3-Clause).

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals



class MyLBFGS(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
        precond_fn=None
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super(MyLBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        logger.inv_h1(f"[Backward] [LBFGS] started.",pad = "************************************************************")
        self.precond_fn = precond_fn
        self.history = []

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure,per_step_max_iter=None):
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        orig_loss = closure()
        loss = float(orig_loss)

        if len(self.history) == 0:
            self.history.append({
                "n_iter": 0,
                "loss": loss,
                "params": [p.detach().clone() for p in self._params],
            })

        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        if opt_cond:
            logger.inv_body(
                f"Gradient tolerance satisfied: {flat_grad.abs().max()} <= {tolerance_grad}"
            )
            return orig_loss

        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        local_iter = 0
        while n_iter < max_iter:
            if per_step_max_iter is not None and local_iter >= per_step_max_iter:
                break

            local_iter += 1
            n_iter += 1
            state['n_iter'] += 1

            if state['n_iter'] == 1:
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1.0

                if self.precond_fn is None:
                    d = flat_grad.neg()
                else:
                    d = -self.precond_fn(flat_grad)

            else:
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = ys / y.dot(y)

                num_old = len(old_dirs)
                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                if self.precond_fn is None:
                    r = q.mul(H_diag)
                else:
                    r = self.precond_fn(q).mul(H_diag)

                d = r
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            if state['n_iter'] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            gtd = flat_grad.dot(d)

            if not torch.isfinite(gtd):
                raise RuntimeError(f"[LBFGS] non-finite directional derivative: gtd={gtd}")

            if gtd >= 0:
                msg = (
                    "[LBFGS] Search direction is not a descent direction. "
                    f"gtd={gtd.item():.6e}. "
                    "This may indicate an invalid preconditioner or corrupted L-BFGS history."
                )
                logger.inv_body(msg, indent=8)

                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1.0

                if self.precond_fn is None:
                    d = flat_grad.neg()
                else:
                    d = -self.precond_fn(flat_grad)

                gtd = flat_grad.dot(d)

                if gtd >= 0:
                    raise RuntimeError(
                        f"[LBFGS] Fallback direction is still not descent: gtd={gtd.item():.6e}"
                    )


            ls_func_evals = 0
            if line_search_fn is not None:
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("MyLBFGS only supports line_search_fn='strong_wolfe'.")
                x_init = self._clone_param()

                def obj_func(x, t_, d_):
                    return self._directional_evaluate(closure, x, t_, d_)

                loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                    obj_func, x_init, t, d, loss, flat_grad, gtd,
                    tolerance_change=tolerance_change
                )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                self._add_grad(t, d)
                if n_iter != max_iter:
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            self.history.append({
                "n_iter": state['n_iter'],
                "loss": loss,
                "params": [p.detach().clone() for p in self._params],
            })
            logger.inv_body(f"LBFGS Iteration: {state['n_iter']}, loss: {loss}")

            if n_iter == max_iter:
                msg = f"[Backward][LBFGS] Breaking out: Reached max_iter = {max_iter}"
                logger.inv_body(msg, indent=8)
                break

            if current_evals >= max_eval:
                msg = f"[Backward][LBFGS] Breaking out: Reached max_eval = {max_eval}"
                logger.inv_body(msg, indent=8)
                break

            if opt_cond:
                msg = "[Backward][LBFGS] Breaking out: Optimal condition satisfied (gradient tolerance)."
                logger.inv_body(msg, indent=8)
                break

            if d.mul(t).abs().max() <= tolerance_change:
                step_size = d.mul(t).abs().max()
                msg = (
                    "[Backward][LBFGS] Breaking out: Step size tolerance reached. "
                    f"Max step size = {step_size}, Tolerance = {tolerance_change}"
                )
                logger.inv_body(msg, indent=8)
                break

            if abs(loss - prev_loss) < tolerance_change:
                msg = (
                    "[Backward][LBFGS] Breaking out: Loss change below tolerance. "
                    f"Loss diff = {abs(loss - prev_loss)}, Tolerance = {tolerance_change}"
                )
                logger.inv_body(msg, indent=8)
                break

            if math.isnan(loss):
                msg = "[Backward][LBFGS] Breaking out: Loss became NaN."
                logger.inv_body(msg, indent=8)
                break


        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return loss


from typing import Dict, List, Sequence, Tuple, Union, Optional, Any

ScaleType = Union[
    float,
    int,
    torch.Tensor,
    Dict[str, Any],
]

def make_block_diag_precond(
    block_specs: Sequence[Tuple[str, int]],
    scales: Dict[str, ScaleType],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):

    if not block_specs:
        raise ValueError("block_specs cannot be empty.")
    total_numel = sum(n for _, n in block_specs)
    if total_numel <= 0:
        raise ValueError("block length must be positive")
    for name, n in block_specs:
        if n <= 0:
            raise ValueError(f"Length of block {name} must be > 0, got {n}")
        if name not in scales:
            raise KeyError(f"Missing scale factor for block {name} in 'scales'.")

    slices: List[Tuple[str, slice, int]] = []
    start = 0
    for name, n in block_specs:
        slices.append((name, slice(start, start + n), n))
        start += n

    def _to_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(x, device=ref.device, dtype=ref.dtype)

    def _apply_block_rule(rule: ScaleType, g_block: torch.Tensor, block_name: str, block_len: int) -> torch.Tensor:
        if not isinstance(rule, dict):
            s = _to_tensor_like(rule, g_block)
            if s.ndim == 0:
                return s * g_block
            if s.ndim == 1 and s.numel() == block_len:
                return s * g_block
            raise ValueError(
                    f"Scale for block '{block_name}' must be a scalar or a 1D tensor "
                    f"of length {block_len}, but got shape={tuple(s.shape)}"
                )

        alpha = rule.get("alpha", 1.0)
        K = rule.get("K", None)
        apply_fn = rule.get("apply", None)
        alpha_t = _to_tensor_like(alpha, g_block)

        if apply_fn is None:
            return alpha_t * g_block

        if not callable(apply_fn):
            raise TypeError(f"'apply' for block '{block_name}' must be callable.")

        out = apply_fn(K, g_block)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                    f"The 'apply' method of block '{block_name}' must return a torch.Tensor, "
                    f"but got {type(out).__name__}."
                )
        if out.shape != g_block.shape:
            raise TypeError(
                    f"The 'apply' method of block '{block_name}' must return a torch.Tensor, "
                    f"but got {type(out).__name__}."
                )
        return alpha_t * out

    def R0(g_flat: torch.Tensor) -> torch.Tensor:
        if g_flat.ndim != 1:
            raise ValueError(f"g_flat must be a 1D tensor, but got shape={tuple(g_flat.shape)}")
        if g_flat.numel() != total_numel:
            raise ValueError(
                f"Size mismatch between g_flat and block_specs: got {g_flat.numel()}, expected {total_numel}"
            )
        out_blocks = []
        for name, s, block_len in slices:
            g_block = g_flat[s]
            out_block = _apply_block_rule(scales[name], g_block, name, block_len)
            out_blocks.append(out_block)
        return torch.cat(out_blocks, dim=0)

    return R0
