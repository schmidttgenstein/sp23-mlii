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

class FCNetTo(MLPipeline):
    def __init__(self,params:list=None,dims:list=None, epoch:int= 250, lr = 0.05,):
        super().__init__(epochs = epoch, lr = lr )
        if dims is not None:
            self.dims = dims 
        elif params is not None:
            weights = params[0]
            bias = params[1]
            dims = [weights[0].shape[1]]
            for w in weights: 
                dims.append(w.shape[0])
        od = OrderedDict()
        self.activation = nn.Sigmoid() 
        for j in range(len(dims)-1):
            od[f"linear_{j}"] = nn.Linear(dims[j],dims[j+1])
            if params is not None:
                od[f"linear_{j}"].weight = nn.Parameter(torch.tensor(weights[j],requires_grad = True).float())
                od[f"linear_{j}"].bias = nn.Parameter(torch.tensor(bias[j],requires_grad = True).float())
            od[f"activation_{j}"] = self.activation

        self.forward_stack = nn.Sequential(od)

    def forward(self,x):
        return self.forward_stack(torch.transpose(x,0,1))

class FCNetFS(MLPipeline):
    def __init__(self,params:list=None, dims:list=None,epochs:int = 1500,lr = .01,):
        super().__init__(epochs = epochs,lr = lr)
        if params is None:
            weights = []
            bias = []
            for j in range(int(len(dims)-1)):
                w = 10 * (np.random.random([dims[j+1],dims[j]])-.5)
                b = 10 * (np.random.random(dims[j+1]) - .5)
                weights.append(w)
                bias.append(b)
        else:
            weights = params[0]
            bias = params[1]
            self.weights = weights 
            self.bias = bias
            dims = [weights[0].shape[1]]
            for w in weights: 
                dims.append(w.shape[0])
        self.weights = weights 
        self.bias = bias 
        self.dims = dims 
        self.n_layers = len(dims)

    def forward(self,x):
        la = self.full_forward(x)
        a = la[-1]
        return a

    def full_forward(self,x):
        assert x.shape[0] == self.dims[0], "Incorrect input dimension!"
        j = 0
        a = x 
        layer_acts = [a]
        for weight in self.weights:
            z_pre = weight @  a
            z = z_pre + self.bias[j].reshape([self.bias[j].shape[0],1])
            if j < self.n_layers - 1:
                a = self.activation(z)
            else: 
                a = z
            layer_acts.append(a)
            j+=1
        return layer_acts

    def backward(self,x_in,y_truth):
        return None

    def update(self,grad):
        return None


    def metrics(self,x,y):
        return None

    def activation(self,z):
        return self.sigmoid(z)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def loss(self,x_in,y_truth):
        ### calculating loss using partial as initial computation
        return None

    def l_grad(self,x_in,y_truth):
        ### partial loss / partial y_pred 
        return None

if __name__ == "__main__":
    #Step 0: No errors here, insantiate a network and "look at it" (you can skip)
    onet = FCNetFS(dims = [1,2,3,1])
    oparams = {'w':onet.weights,'b':onet.bias, 'd':onet.dims}
    print(oparams)
    
    #Step 1: Generate "From Scratch" Network and Instantiate Torch Network with these weights
    anet = FCNetFS(dims = [1,5,3,5,1])
    in_weights = anet.weights
    in_bias = anet.bias
    in_dim = anet.dims
    bnet = FCNetTo(params=[in_weights,in_bias])
    
    #Before continuing, examine your bnet.forward_stack to ensure that layers are coordinated as you 
    n_samp = 350
    tspace = np.linspace(-15,15,n_samp).reshape([1,n_samp])
    y1 =  anet.forward(tspace)
    tensor_tsp = torch.tensor(tspace)
    y2 = bnet.forward(tensor_tsp.float())
    
    #Step 1.5 sanity check: can you reproduce From Scratch network with prespecified weights params?
    a2net = FCNetFS(params=[in_weights,in_bias])
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
    for layer in cnet.forward_stack:
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
    print("THANKS")