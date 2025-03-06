import numpy as np
from tensor import Tensor

class Add:
    @staticmethod
    def forward(ctx, a:Tensor, b:Tensor):
        result_data = a.data + b.data
        ctx['a_shape'] = a.data.shape
        ctx['b_shape'] = b.data.shape
        result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
        result.grad_fn = Add
        result.ctx = ctx
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_a = grad_output
        grad_b = grad_output
        return grad_a, grad_b
    

class ReLU:
    @staticmethod
    def forward(ctx, a:Tensor):
        result_data = np.array([max(cell, 0) for cell in a.data.flat]).reshape(a.data.shape)
        ctx['mask'] = (a.data >= 0)
        ctx['shape'] = a.data.shape
        result = Tensor(result_data, requires_grad=a.requires_grad)
        result.grad_fn = ReLU
        result.ctx = ctx
        return result

    def backward(ctx, grad_output):
        grad_a = np.array([cell if ctx['mask'].flat[idx] else 0 for idx, cell in enumerate(grad_output)]).reshape(grad_output.shape)
        return grad_a
    
class MatMul:
    @staticmethod
    def forward(ctx, a:Tensor, b:Tensor):
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        pass

class SoftMax:
    @staticmethod
    def forward(ctx, a:Tensor):
        pass
    
    @staticmethod
    def backward(ctx, a:Tensor):
        pass

ctx = {}
a = Tensor([-1, 2], requires_grad=True)
b = Tensor([3, 4], requires_grad=True)
c = Add.forward(ctx, a, b)
d = ReLU.forward(ctx, a)

print(d.backward())