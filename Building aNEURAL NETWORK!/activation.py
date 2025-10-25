
import numpy as np

class Relu:
    def __call__(self,pre_activated_output):
        return np.where(pre_activated_output<=0,0,1) * pre_activated_output
class Sigmoid:
    def __call__(self,pre_activated_output):
        return 1+(1.np.exp(-pre_activated_output))
class Softmax:
    def __call__(self,pre_activated_output):