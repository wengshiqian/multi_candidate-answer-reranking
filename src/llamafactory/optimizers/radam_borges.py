import math
import torch
from torch.optim.optimizer import Optimizer


class RAdamBorges(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 borges_smoothing=1, min_gradient_scale=0.3, max_gradient_scale=2.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        borges_smoothing=borges_smoothing,
                        min_gradient_scale=min_gradient_scale,
                        max_gradient_scale=max_gradient_scale)
        super(RAdamBorges, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['borges_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, borges_grad = state['exp_avg'], state['exp_avg_sq'], state['borges_grad']
                beta1, beta2 = group['betas']
                borges_smoothing = group['borges_smoothing']
                min_gradient_scale = group['min_gradient_scale']
                max_gradient_scale = group['max_gradient_scale']

                state['step'] += 1
                step = state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Borges 自适应梯度平滑
                # borges_smoothing_adaptive = borges_smoothing * (1 - torch.sigmoid(torch.norm(grad)))
                borges_smoothing_adaptive = borges_smoothing * (1 - torch.tanh(torch.norm(grad)))
                borges_smoothing_adaptive = torch.clamp(borges_smoothing_adaptive, min=0.1, max=0.9)
                borges_grad.mul_(borges_smoothing_adaptive).add_(grad, alpha=1 - borges_smoothing_adaptive)

                # 梯度归一化
                grad_std = torch.std(borges_grad) + 1e-8
                improved_grad = borges_grad / grad_std

                # 计算动量
                exp_avg.mul_(beta1).add_(improved_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(improved_grad, improved_grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** step

                # 动态梯度缩放
                gradient_scale_factor = torch.clamp(1 / grad_std, min_gradient_scale, max_gradient_scale)
                scaled_grad = improved_grad * gradient_scale_factor

                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * beta2 ** step / (1 - beta2 ** step)

                # 参数更新
                if rho_t > 5:
                    adaptive_lr = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    step_size = group['lr'] * adaptive_lr / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(-step_size, scaled_grad, denom)
                else:
                    # 原来: p.data.add_(-group['lr'], scaled_grad)
                    p.data.add_(scaled_grad, alpha=-group['lr'])

        return loss
        # if N_sma >= 5:
        #     if group['weight_decay'] != 0:
        #         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
        #     denom = exp_avg_sq.sqrt().add_(group['eps'])
        #     p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
        #     p.data.copy_(p_data_fp32)
        # elif step_size > 0:
        #     if group['weight_decay'] != 0:
        #         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
        #     p_data_fp32.add_(-step_size * group['lr'], exp_avg)
        #     # p.data.copy_(p_data_fp32)