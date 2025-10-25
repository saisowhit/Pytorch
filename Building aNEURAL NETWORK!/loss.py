
import numpy as np

class Relu:
    def __call__(self,pre_activated_output):
        return np.maximum(pre_activated_output,0)
    def derivative(self,output,grad_so_far):
        return np.where(output<=0,0,1)*grad_so_far
class sigmoid:
    def __call__(self, pre_activated_output):
        pre_activated_output=np.clip(pre_activated_output,-1000,1000)
        return 1/(1+np.exp(-pre_activated_output))
    def derivative(self,output,grad_so_far):
        return (1/1(1+np.exp(-pre_activated_output)))*(1-(1/(1+np.exp(-pre_activated_output)))*grad_so_far
class Softmax:
    def __call__(self,pre_activated_output):
        exp_shifted=np.exp(pre_activated_output-np.max(pre_activated_output,axis=1,keepdims=True))
        denominator=np.sum(exp_shifted)
        return exp_shifted/denominator
    def derivative(self,pre_actviated_output,grad_so_far):