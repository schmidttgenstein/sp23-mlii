import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import numpy.linalg as la 
import matplotlib.pyplot as plt 
from collections import OrderedDict



class MLPipeline:
    def __init__(self,epochs = 250,lr = 0.025):
        ###In this constructor we set the model complexity, number of epochs for training, 
        ##and learning rate lr. You should think of complexity here as "number of parameters"
        #defining model. In linear regression, this e.g. may be (deg of poly)-1. 
        self.epochs = epochs
        self.lr = lr 

    def gen_data(self,):
        raise NotImplementedError

    def loss(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError

    def backward(self,):
        raise NotImplementedError 

    def update(self,):
        raise NotImplementedError  

    def metrics(self,x,y):
        raise NotImplementedError     

    def fit(self,x_data,y_data,x_eval,y_eval, printing = False):
        ### This method implements our "1. forward 2. backward 3. update paradigm"
        ## it should call forward(), grad(), and update(), in that order. 
        # you should also call metrics so that you may print progress during training
        if printing:
            self.x_eval = x_eval 
            self.y_eval = y_eval
        for epoch in range(self.epochs):
            y_pred = self.forward(x_data)
            loss = self.loss(y_pred,y_data)
            grad = self.backward(x_data,y_data)
            self.update(grad)
            if printing: 
                m = self.metrics(x_eval,y_eval)
                if epoch % 100 == 0:
                    print(f"epoch {epoch} and train loss {loss.mean():.2f}, test metrics {m:.2f}")
        if printing:
            self.m = m

class FCNetFS(MLPipeline):
    def __init__(self, params:list = None, dims:list=None,epochs:int = 1500,lr = .01): #dims can not be none when implementing
        super().__init__()
        
        if params is None:
            weights = []
            bias = []
            for j in range(len(dims) - 1):
                w = 10 * ((np.random.random([dims[j+1], dims[j]])) - 0.5)
                b = 10 * (np.random.random(dims[j+1]) - 0.5)
                weights.append(w)
                bias.append(b)
        
        else: #params is not none, params = [weight_lists of list, bias lists of list]
            weights = params[0]
            bias = params[1]
            # the first weight
            d = (np.array(weights[0])).shape[1]
            dims = [d]
            
            for w in weights:
                d = (np.array(w)).shape[0]
                dims.append(d)
            
        self.weights = weights
        self.bias = bias
        self.dims = dims
        self.num_layer = len(dims)
        self.epochs = epochs
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
        
    def full_forward(self, x):
        # make sure x dimension is correct
        if x.shape[0] != 1:
            assert x.shape[0] == (self.weights[0]).shape[0], "x input wrong dimension"
        neurons = [x]
        j = 0
        for w in self.weights:
            a = neurons[j]
            b = self.bias[j]
            #print(b.shape)
            z = (w @ a) + b.reshape([b.shape[0], 1])
            #e.g
            # x = [x1,.....x350]
            # w @ x = 5 by 350
            # b is 5 by 1
            
            
            if j < self.num_layer - 1:
                a_new = self.sigmoid(z)
            else:
                a_new = z
                
            neurons.append(a_new)
            j = j + 1
            
        return neurons
    
    def forward(self,x): #return the last layer
        la = self.full_forward(x)
        a = la[-1]
        return a

class FCNetTo(MLPipeline):
    def __init__(self, params:list=None,dims:list=None, epoch:int= 250, lr = 0.05):
        super().__init__(epochs = epoch, lr = lr)
        
        self.activation = nn.Sigmoid()
        od = OrderedDict()
        
        if dims is not None:
            self.dims = dims
        if params is not None:
            weights = params[0]
            bias = params[1]
            dims = [weights[0].shape[1]]
            for w in weights:
                d = w.shape[0]
                dims.append(d)
                
        for j in range(len(dims) - 1): #prepare to pass into sequential to construct the model
            od[f"linear_{j}"] = nn.Linear(dims[j], dims[j+1])
            
        
            if params is not None: #update the weight and bias
                od[f"linear_{j}"].weight = nn.Parameter(torch.tensor(weights[j], requires_grad = True).float())
                od[f"linear_{j}"].bias = nn.Parameter(torch.tensor(bias[j], requires_grad = True).float())
            od[f"activation_{j}"] = self.activation
            
        self.model = nn.Sequential(od)
        
    def forward(self, x):
        #The given dimensions dim0 and dim1 are swapped.
        r = self.model(torch.transpose(x,0,1)) #prepare the x to be inputed
        return r
         





if __name__ == "__main__":
    #Step 0: No errors here, insantiate a network and "look at it" (you can skip)
    onet = FCNetFS(dims = [1,2,3,1])
    onet.weights
    onet.bias 
    onet.dims 
    
    #Step 1: Generate "From Scratch" Network and Instantiate Torch Network with these weights
    anet = FCNetFS(dims = [1,5,3,5,1],epochs = 250, lr = 0.01)
    in_params = [anet.weights, anet.bias]
    bnet = FCNetTo(params=in_params)
    #Before continuing, examine your bnet.forward_stack to ensure that layers are coordinated as you 
    n_samp = 350
    tspace = np.linspace(-15,15,n_samp).reshape(1,n_samp)
    y1 =  anet.forward(tspace)
    y2 = bnet.forward(torch.tensor(tspace).float())
    #Step 1.5 sanity check: can you reproduce From Scratch network with prespecified weights params?
    a2net = FCNetFS(params=in_params)
    y3 = a2net.forward(tspace) 
    plt.figure(1)
    plt.plot(tspace.flatten(),y1.flatten())
    plt.plot(tspace.flatten(),y2.detach().numpy(),'--')
    plt.figure(2)
    plt.plot(tspace.flatten(),np.abs(y1.flatten()-y2.detach().numpy().flatten())+np.abs(y1.flatten()-y3.flatten()))
    

    #Step 2: Generate "To[rch]" Network and Instantiate from scratch network with these weights
    cnet = FCNetTo(dims = [1,7,5,7,1])
    weights = []
    bias = []
    for layer in cnet.model:
        if isinstance(layer,nn.Linear):
            weights.append(layer.weight.detach().numpy())
            bias.append(layer.bias.detach().numpy())
    params = [weights,bias]
    dnet = FCNetFS(params)

    y3 =  cnet.forward(torch.tensor(tspace).float())
    y4 = dnet.forward(tspace)
    plt.figure(3)
    plt.plot(tspace.flatten(),y4.flatten())
    plt.plot(tspace.flatten(),y3.detach().numpy(),'--')
    plt.figure(4)
    plt.plot(tspace.flatten(),np.abs(y4.flatten()-y3.detach().numpy().flatten()))
    plt.show()