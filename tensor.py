import numpy as np

class Tensor:
    def __init__(self, data, requires_grad = True, grad_fn = None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn
        
    def backward(self, grad = None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad
        
        if self.grad_fn:
            self.grad_fn.backward(self.ctx, self.grad)
            