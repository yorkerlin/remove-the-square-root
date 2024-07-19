import torch
import torch.optim as optim


class MyRmsProp(optim.Optimizer):
    #root-free RMSProp
    def __init__(self, params, lr, alpha, eps=1e-4,
            weight_decay=0, momentum=0,
            batch_averaged=True,
            batch_size=None,
            cast_dtype=torch.float32,
            model=None,
            dummy_init = False,
            dummy_scaling = False,
            ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        print('rf-rmsprop', cast_dtype)
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, weight_decay=weight_decay)

        self.dummy_init=dummy_init
        if self.dummy_init:
            print( 'enable zero init')

        self.dummy_scaling=dummy_scaling
        if self.dummy_scaling:
            print( 'enable default scaling')

        self.cast_dtype = cast_dtype
        self.batch_averaged = batch_averaged
        if batch_averaged:
            assert batch_size is not None
        self.batch_size = batch_size
        super(MyRmsProp, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
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
                grad = p.grad.to(self.cast_dtype)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if self.dummy_init:
                        state['square_avg'] = torch.zeros_like(grad)
                    else:
                        state['square_avg'] = torch.ones_like(grad)/factor

                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(grad)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

###################################################################
                lr_cov = 1.0-alpha
                lr0 = group['lr']

                square_avg.mul_(1.0-lr_cov).addcmul_(grad, grad, value=lr_cov)
###################################################################
                grad.div_( (square_avg*factor + group['eps']) )

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad)
                else:
                    buf = grad

                p.add_(buf, alpha=-lr0)

        return loss
