import torch 

class GlobalAdam(torch.optim.Adam):
    # param: model parameters that the optimizer will optimize
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Track the number of optimization steps for each parameters
                state['step'] = 0
                # Store the exponential moving average of past gradients (bias correction)
                state['exp_avg'] = torch.zeros_like(p.data)
                # Store the exponential moving average of squared gradients (scale learning rate adatively)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # Allow multiple processes to access and modify
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                