import math

import numpy as np
import torch
import torch.optim as optim

class LocalOptimizer_GGT(optim.Optimizer):
    #Root and inverse-free Shampoo

    def __init__(
        self,
        model,
        lr=0.001,
        momentum=0.9,
        damping=0.001,
        beta2=0.5, #Riemannian momentum
        weight_decay=0.0,
        T=10,
        batch_averaged=True,
        lr_cov=1e-2, #weight to update preconditioner factors
        batch_size=None,
        cast_dtype = torch.float32,
    ):
        print("IF-Shampoo")
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_cov < 0.0:
            raise ValueError("Invalid learning rate for cov: {}".format(lr_cov))
        if beta2 < 0.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.lr_cov = lr_cov
        self.damping = damping
        self.beta2 = beta2
        self.cast_dtype = cast_dtype

        self.batch_averaged = batch_averaged
        if batch_averaged:
            assert batch_size is not None
            self.batch_size = batch_size

        self.params_list = {}
        params = self._prepare_model(model)
        super(LocalOptimizer_GGT, self).__init__(params, defaults)

        self.steps = 0
        self.next_step = 0
        self.scaling = {}

        self.precond_B_blocks = {}
        self.precond_m_B_blocks = {}
        self.precond_BBt_blocks = {}
        self.T = T


    def _prepare_model(self, model):  #ok
        named_params = model.named_parameters()
        params = []
        for name, param in named_params:
            if param.requires_grad:
                key = name
                info = self.params_list.setdefault(key, [])
                info.append(param)
                params.append(param)
        return params

    def get_H_values(self, key, G, precond_B_blocks, damping):
        results = {}

        #we assume that G = torch.squeeze(p_grad)
        if len(G.shape)==1:
            name = '%s_dim-%d'%(key, 1)
            d1 = precond_B_blocks[name].shape[0]

            results[name] = (damping/2.0 * precond_B_blocks[name].t()) @ precond_B_blocks[name]
            half = ( G.view(1, -1) ) @ precond_B_blocks[name]
            results[name].add_(half.t() @ half, alpha=1.0/2.0)

            k = torch.tensor(range(d1))
            results[name][k, k] = torch.diagonal(results[name]) - 1.0/2.0

        elif len(G.shape)==2:
            name1 = '%s_dim-%d'%(key, 1)
            name2 = '%s_dim-%d'%(key, 2)
            d1 = precond_B_blocks[name1].shape[0]
            d2 = precond_B_blocks[name2].shape[0]

            B1 = precond_B_blocks[name1]/math.sqrt(d1)
            B2 = precond_B_blocks[name2]/math.sqrt(d2)

            results[name1] = B1.t() @ B1
            results[name2] = B2.t() @ B2
            tr_BBt1 = torch.trace(results[name1])
            tr_BBt2 = torch.trace(results[name2])
            results[name1].mul_((damping*d1)*tr_BBt2/2.0)
            results[name2].mul_((damping*d2)*tr_BBt1/2.0)

            tmp = B2.t() @ G @ B1
            results[name1].add_(tmp.t() @ tmp, alpha=d1/2.0)
            results[name2].add_(tmp @ tmp.t(), alpha=d2/2.0)


            k = torch.tensor(range(d1))
            results[name1][k, k] = torch.diagonal(results[name1]) - 1.0/2.0

            k = torch.tensor(range(d2))
            results[name2][k, k] = torch.diagonal(results[name2]) - 1.0/2.0

        elif len(G.shape)==3: #3d tensor
            name1 = '%s_dim-%d'%(key, 1)
            name2 = '%s_dim-%d'%(key, 2)
            name3 = '%s_dim-%d'%(key, 3)

            d1 = precond_B_blocks[name1].shape[0]
            d2 = precond_B_blocks[name2].shape[0]
            d3 = precond_B_blocks[name3].shape[0]

            B1 = precond_B_blocks[name1]/math.sqrt(d1)
            B2 = precond_B_blocks[name2]/math.sqrt(d2)
            B3 = precond_B_blocks[name3]/math.sqrt(d3)
            results[name1] = B1.t() @ B1
            results[name2] = B2.t() @ B2
            results[name3] = B3.t() @ B3
            tr_BBt1 = torch.trace( results[name1] )
            tr_BBt2 = torch.trace( results[name2] )
            tr_BBt3 = torch.trace( results[name3] )
            results[name1].mul_( (d1*damping)*tr_BBt2*tr_BBt3/2.0  )
            results[name2].mul_( (d2*damping)*tr_BBt1*tr_BBt3/2.0  )
            results[name3].mul_( (d3*damping)*tr_BBt1*tr_BBt2/2.0  )

            tmp_common = torch.einsum('pi,ijk->pjk',  B3.t(), G)
            tmp1_half = torch.einsum('pjk,jq->pqk', tmp_common, B2)
            tmp11 = torch.einsum( 'pqk,ku->pqu', tmp1_half, precond_B_blocks[name1]).view(-1,d1)
            results[name1].add_(tmp11.t() @ tmp11, alpha=1.0/2.0)



            tmp2_half = torch.einsum('pjk,km->pjm', tmp_common, B1)
            tmp22 = torch.einsum( 'pjm,ju->pmu', tmp2_half, precond_B_blocks[name2]).view(-1,d2)
            results[name2].add_(tmp22.t() @ tmp22, alpha=1.0/2.0)


            tmp_remaining = torch.einsum('ijk,jq->iqk',  G,  B2)
            tmp3_half = torch.einsum( 'iqk,km->iqm', tmp_remaining, B1)
            tmp33 = torch.einsum( 'iqm,iu->qmu', tmp3_half, precond_B_blocks[name3]).view(-1,d3)
            results[name3].add_(tmp33.t() @ tmp33, alpha=1.0/2.0)

            k = torch.tensor(range(d1))
            results[name1][k, k] = torch.diagonal(results[name1]) - 1.0/2.0

            k = torch.tensor(range(d2))
            results[name2][k, k] = torch.diagonal(results[name2]) - 1.0/2.0

            k = torch.tensor(range(d3))
            results[name3][k, k] = torch.diagonal(results[name3]) - 1.0/2.0

        elif len(G.shape)==4: #4d tensor
            name1 = '%s_dim-%d'%(key, 1)
            name2 = '%s_dim-%d'%(key, 2)
            name3 = '%s_dim-%d'%(key, 3)
            name4 = '%s_dim-%d'%(key, 4)
            d1 = precond_B_blocks[name1].shape[0]
            d2 = precond_B_blocks[name2].shape[0]
            d3 = precond_B_blocks[name3].shape[0]
            d4 = precond_B_blocks[name4].shape[0]


            B1 = precond_B_blocks[name1]/math.sqrt(d1)
            B2 = precond_B_blocks[name2]/math.sqrt(d2)
            B3 = precond_B_blocks[name3]/math.sqrt(d3)
            B4 = precond_B_blocks[name4]/math.sqrt(d4)
            results[name1] = B1.t() @ B1
            results[name2] = B2.t() @ B2
            results[name3] = B3.t() @ B3
            results[name4] = B4.t() @ B4
            tr_BBt1 = torch.trace( results[name1] )
            tr_BBt2 = torch.trace( results[name2] )
            tr_BBt3 = torch.trace( results[name3] )
            tr_BBt4 = torch.trace( results[name4] )
            results[name1].mul_( (d1*damping)*tr_BBt2*tr_BBt3*tr_BBt4/2.0  )
            results[name2].mul_( (d2*damping)*tr_BBt1*tr_BBt3*tr_BBt4/2.0  )
            results[name3].mul_( (d3*damping)*tr_BBt1*tr_BBt2*tr_BBt4/2.0  )
            results[name4].mul_( (d4*damping)*tr_BBt1*tr_BBt2*tr_BBt3/2.0  )


            tmp_common = torch.einsum('pi,ijkl->pjkl',  B4.t(), G)
            tmp_a = torch.einsum( 'pjkl,jq->pqkl', tmp_common, B3)
            tmp1_half = torch.einsum('pqkl,km->pqml', tmp_a, B2)
            tmp11 = torch.einsum('pqml,lu->pqmu', tmp1_half, precond_B_blocks[name1]).view(-1,d1)
            results[name1].add_(tmp11.t() @ tmp11, alpha=1.0/2.0)

            tmp2_half = torch.einsum('pqkl,lw->pqkw', tmp_a, B1)
            tmp22 = torch.einsum( 'pqkw,ku->pqwu', tmp2_half, precond_B_blocks[name2]).view(-1,d2)
            results[name2].add_(tmp22.t() @ tmp22, alpha=1.0/2.0)

            tmp_b = torch.einsum( 'pjkl,km->pjml', tmp_common, B2)
            tmp3_half = torch.einsum('pjml,lw->pjmw', tmp_b, B1)
            tmp33 = torch.einsum( 'pjmw,ju->pmwu', tmp3_half, precond_B_blocks[name3]).view(-1,d3)
            results[name3].add_(tmp33.t() @ tmp33, alpha=1.0/2.0)

            tmp_remaining =  torch.einsum('ijkl,jq->iqkl',  G, B3)
            tmp_c = torch.einsum( 'iqkl,km->iqml', tmp_remaining, B2)
            tmp4_half = torch.einsum('iqml,lw->iqmw', tmp_c, B1)
            tmp44 = torch.einsum( 'iqmw,iu->qmwu', tmp4_half, precond_B_blocks[name4]).view(-1,d4)
            results[name4].add_(tmp44.t() @ tmp44, alpha=1.0/2.0)


            k = torch.tensor(range(d1))
            results[name1][k, k] = torch.diagonal(results[name1]) - 1.0/2.0

            k = torch.tensor(range(d2))
            results[name2][k, k] = torch.diagonal(results[name2]) - 1.0/2.0

            k = torch.tensor(range(d3))
            results[name3][k, k] = torch.diagonal(results[name3]) - 1.0/2.0

            k = torch.tensor(range(d4))
            results[name4][k, k] = torch.diagonal(results[name4]) - 1.0/2.0

        else:
            raise NotImplementedError

        return results


    def _update_inv(self, m, G):
        """compute the inverse of the preconditioner.
        :param m: The layer
        :return: no returns.
        """
        group = self.param_groups[0]
        damping = self.damping

        res = self.get_H_values(m, G, self.precond_B_blocks, damping)
        self.scaling[m] = 1.0
        lr1 = self.lr_cov

        if m.find('norm')>=0 or m.find('proj')>=0 or m.find('bias')>=0:
            # if self.steps==0: print(m, 'discounted')
            lr1 = lr1 * 0.25

        beta2 = self.beta2
        total_dim = len(G.shape)
        for idx, dim in enumerate(G.shape):
            name = '%s_dim-%d'%(m, total_dim - idx)
            if total_dim>1: assert dim>1
            if dim==1: assert total_dim == 1

            assert self.precond_m_B_blocks[name].shape ==  res[name].shape
            self.precond_m_B_blocks[name].mul_(beta2).add_(
                        res[name], alpha=(1.0 - beta2)
                    )  # beta2: alpha_1 in the paper  (riemannian momentum for K)


            #using a matrix norm to update the factors
            norm_ = torch.norm(self.precond_m_B_blocks[name])
            norm_B = torch.max(torch.tensor([1.0, norm_]))

            self.precond_B_blocks[name].add_(
               (self.precond_B_blocks[name] @ self.precond_m_B_blocks[name]), alpha=-lr1 / norm_B
            )  # lr1:beta_1 in the paper  (first-order truncation for the expm)


            scaling_B = self.get_scaling(self.precond_B_blocks[name])
            tmp_B = self.precond_B_blocks[name]/scaling_B
            self.precond_BBt_blocks[name] = tmp_B @ (tmp_B.t())
            assert torch.isfinite(self.precond_BBt_blocks[name]).all()
            self.scaling[m] *= scaling_B**2



    def _group_param_grad(self, block, key, cast_dtype=torch.float32):
        assert len(block) == 1
        W = torch.squeeze(block[0].grad)
        assert W is not None
        if len(W.shape)>2 and self.steps==0: print(W.shape, key)

        assert torch.isfinite(W).all()

        return W.to(dtype=cast_dtype)


    def _update_natural_grad(self, m, block,  p_grad, damping):
        """
        :param m:  the layer
        :param p_grad: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        total_dim = len(p_grad.shape)
        if total_dim == 1: #1d tensor (vector)
            name = '%s_dim-%d'%(m, 1)
            v = self.precond_BBt_blocks[name] @ p_grad
        elif total_dim == 2: #2d tensor (matrix)
            name1 = '%s_dim-%d'%(m, 1)
            name2 = '%s_dim-%d'%(m, 2)
            v = self.precond_BBt_blocks[name2] @ p_grad @ self.precond_BBt_blocks[name1]
        elif total_dim == 3:
            name1 = '%s_dim-%d'%(m, 1)
            name2 = '%s_dim-%d'%(m, 2)
            name3 = '%s_dim-%d'%(m, 3)
            v = torch.einsum('pi,ijk->pjk', self.precond_BBt_blocks[name3], p_grad)
            v = torch.einsum('pjk,jq->pqk', v, self.precond_BBt_blocks[name2])
            v = torch.einsum('pqk,km->pqm', v, self.precond_BBt_blocks[name1])
        elif total_dim == 4:
            name1 = '%s_dim-%d'%(m, 1)
            name2 = '%s_dim-%d'%(m, 2)
            name3 = '%s_dim-%d'%(m, 3)
            name4 = '%s_dim-%d'%(m, 4)

            v = torch.einsum('pi,ijkl->pjkl', self.precond_BBt_blocks[name4], p_grad)
            v = torch.einsum('pjkl,jq->pqkl', v, self.precond_BBt_blocks[name3])
            v = torch.einsum('pqkl,km->pqml', v, self.precond_BBt_blocks[name2])
            v = torch.einsum('pqml,lw->pqmw', v, self.precond_BBt_blocks[name1])
        else:
            raise NotImplementedError

        v = v * self.scaling[m]
        assert torch.isfinite(v).all()

        block[0].grad.data.copy_(v.view(block[0].grad.data.size()))

        return v

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data  # grad or natural_grad

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)  # add weight decay into momentum

                if momentum != 0:  # add momentum
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        # buf = param_state["momentum_buffer"] = torch.zeros_like(d_p) #note: this uses float32
                        buf = param_state["momentum_buffer"] = torch.zeros_like(d_p.to(self.cast_dtype))
                    else:
                        buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p)  # add the standard momentum
                    d_p = buf

                assert torch.isfinite(d_p).all()
                p.data.add_(d_p, alpha=-group["lr"])  # perform a SGD-like update



    def get_scaling(self, A):
        _scaling = np.max( [1.0,
             torch.sqrt( torch.max(torch.abs(A)) ).item()
            ] )
        return _scaling


    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        momentum = group["momentum"]
        lr = group["lr"]
        damping = self.damping


        for key, block in self.params_list.items():
            p_grad = self._group_param_grad(block, key, cast_dtype=self.cast_dtype)

            if self.steps == 0:
                self.scaling[key] = 1.0
                total_dim = len(p_grad.shape)
                for idx, dim in enumerate(p_grad.shape):
                    name = '%s_dim-%d'%(key, total_dim - idx)

                    self.precond_B_blocks[name] = torch.diag(p_grad.new(dim).fill_(1))
                    self.precond_m_B_blocks[name] = torch.zeros_like(self.precond_B_blocks[name])
                    self.precond_BBt_blocks[name] = torch.ones_like(self.precond_B_blocks[name])


            # if self.steps % self.T == 0:
            if self.steps == self.next_step:
                factor = 1.0  # since grad is unscaled
                if self.batch_averaged:
                    factor *= math.sqrt(self.batch_size)

                self._update_inv(
                       key,
                       factor * p_grad
                    )  # inverse fim/hessian estimation
 
            self._update_natural_grad(key, block, p_grad, damping)


        if self.steps == self.next_step:
            diff = min(max(int(math.log(self.steps+1, 4)),1), self.T)
            self.next_step = diff + self.steps
            # print('next step is', self.next_step, diff, self.steps)

        self._step(closure)
        self.steps += 1

