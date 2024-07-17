import torch
from torch import optim


class MyAdamW(optim.Optimizer):
    #root-free AdamW

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2,
                batch_averaged=True,
                batch_size=None,
                cast_dtype=torch.float32,
                dummy_init = False,
                dummy_scaling = False,
                bias_correction = False,
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        print('rf-adamw')

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)


        self.dummy_init=dummy_init
        if self.dummy_init:
            print( 'enable dummy init')

        self.dummy_scaling=dummy_scaling
        if self.dummy_scaling:
            print( 'enable dummy scaling')

        self.bias_correction = bias_correction
        if not self.bias_correction:
            print( 'disable bias correction')

        self.cast_dtype = cast_dtype
        self.batch_averaged = batch_averaged
        if batch_averaged:
            assert batch_size is not None
        self.batch_size = batch_size
 

        super(MyAdamW, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        factor = 1.0  # since grad is unscaled
        if self.dummy_scaling:
            factor = 1.0
        else:
            if self.batch_averaged:
                factor *= self.batch_size

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.to(self.cast_dtype)
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values

                    if self.dummy_init:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    else:
                        state['exp_avg_sq'] = torch.ones_like(grad)/factor



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1.0
                bias_correction2 = 1.0
                if self.bias_correction:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq*factor / bias_correction2).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
