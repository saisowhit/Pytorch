
import numpy as np
from activation  import *
from random import shuffle

class Layer:
    def __init__(self,input_size,output_size,alpha,batch_size,bias=False,activation_function=None):
        self.input_size=input_size
        self.output_size=output_size
        self.alpha=alpha
        self.activation_function=activation_function
        self.weights=np.random.rand(self.input_size,self.output_size)
        self.bias=bias
        if bias:
            self.weights=np.vstack(self.weights,np.random.rand(1,self.output_size))
        self.update_matrix=None
        self.current_inputs=None
        self.pre_activated_output=None
        print(self.weights)
    def get_batch_size(self):
        return self.get_batch_size
    def set_alpha(self,new_alpha):
        self.alpha=new_alpha
    def __call__(self,layer_inputs):
        if self.bias:
            layer_inputs=np.hstac(layer_inputs,np.ones((self.batch_size,1)))
        layer_outputs=layer_inputs@ self.weights
        print(layer_outputs)
        if self.activation_func is not None:
            self.pre_activated_output=np.copy(layer_outputs)
            layer_outputs=self.activation_func(layer_outputs)
        return layer_outputs


class LayerList:
    def __init__(self,*layers):
        self.model=list(layers)
    def append(self,*layers):
        for layer in layers:
            self.model.append(layer)
    def set_alpha(self,new_alpha):
        for layer in self.model:
            layer.set_alpha()
    def __call__(self,model_input):
        intermediate_results=model_input
        for layer in self.model:
            intermediate_results=layer(intermediate_results)
        return intermediate_results
    def back(self,ret):
        if self.activation_func is not None:
            ret=self.activation_fun.derivative(self.pre_activated_output,ret)
        self.update_matrix=self.current_inputs@ret
        new_ret=ret@self.weights
        if self.bias:
            new_ret=new_ret[:,:-1]
        return new_ret
    def update(self):
        self.weight-=self.alpha*self.update_matrix
        self.update_matrix
    def back(self,error):
        for layer in self.model[::-1]:
            error=layer.back(error)
    def step(self):
        for layer in self.model:
            layer.update()
    @staticmethod
    def batch(input_data,expected,batch_size):
        num_data=input_data.shape[0]
        indices=[i for i in range(num_data)]
        shuffle(indices)
        batched_inputs,batch_expected=[],[]
        for _ in range(num_data//batch_size):
            batch_inp,batch_exp=[],[]
            for j in range(batch_size):
                batch_inp.append(input_data[indices[batch_size*i+j]])
                batch_exp.append(np.array(batch_expected))

        return np.array(batched_inputs),np.array(batch_expected)
    def fit(self,input_data,expected,epochs,alpha,batch_size):
        self.set_alpha(alpha)
        prev_update=1
        for e in range(epochs):
            batch_input,batch_expected=Layer.batch(input_data,expected,batch_size)
            for i in range(len(batch_input)):
                model_output=self(batch_input)
                self.back(loss_deriv_func(model_output,batch_expected[i]))
                self.step()
            if e==10*prev_update:
                alpha/=10
                self.set_alpha(alpha)
                prev_update=0


        for _ in range(epochs):
            batched_input,batch_expected=LayerList.batch(input_data,expected,batch_size)
            for i  in range(len(batched_input)):
                model_output=self(batched_input)
                self.back(loss_deriv_func(model_output,batch_expected[i]))
                self.step()
    def predict(self,inputs):
        predictions=[]
        for inp in inputs:
            ## (1,num_features)
            predictions.append(self(np.expand_dims(inp)))
        return predictions

if __name__=="__main__":
    model=LayerList(Layer(1,1,0.1,1),Layer(1,2,0.1,1,activation_function=Softmax()))
    inp=np.array([[1]])
    model(inp)