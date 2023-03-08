import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import numpy.linalg as la 
import matplotlib.pyplot as plt 
from collections import OrderedDict


class DataGen:
    def __init__(self,dim = 10,N = 20000):
        self.dim = dim 
        self.N = N 

        
    def gen_data(self,n = None,split = 0.6,mu_factor = 1):
        x0,y0 = self.gen_label_data(label = 0,n_dat = n,mult = mu_factor)
        x1,y1 = self.gen_label_data(label = 1,n_dat = n,mult = mu_factor)
        cutoff = int(split * x0.shape[1])
        x_tr = np.concatenate([x0[:,:cutoff],x1[:,:cutoff]],axis = 1)
        y_tr = np.concatenate([y0[:,:cutoff],y1[:,:cutoff]],axis = 1)
        x_te = np.concatenate([x0[:,cutoff:],x1[:,cutoff:]],axis = 1)
        y_te = np.concatenate([y0[:,cutoff:],y1[:,cutoff:]],axis = 1)
        return x_tr,y_tr,x_te,y_te

    def gen_label_data(self,label = 0, n_dat = None,mult=1):
        if n_dat is None:
            n_dat = self.N 
        y = np.zeros([2,n_dat])
        m = mult * np.random.random(self.dim) - .5 
        C = np.random.random([self.dim,self.dim]) - .5
        C = C @ np.transpose(C)
        x = np.random.multivariate_normal(m, C, n_dat)
        y[label,:] = 1
        return np.transpose(x),y 

    def ym2yo(self,ym):
        vals = 0 * ym
        vals[1,:] = 1
        idx1 = ym == 1
        yo = vals[idx1]
        return yo 

    def probe_dat(self,x,y,coord = 0):
        yo = self.ym2yo(y)
        idx1 = yo == 1 
        x1 = x[:,idx1]
        x0 = x[:,~idx1]
        plt.hist(x0[coord,:],density = True, bins = 100,label = 'x(y=0)')
        plt.hist(x1[coord,:],density = True, bins = 100,label = 'x(y=1)')
        plt.legend()
        # plt.show()




class MLPipeline:
    def __init__(self,epochs = 10,lr = 0.025):
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

    def full_forward(self,):
        raise NotImplementedError 
    
    def backward(self,):
        raise NotImplementedError 

    def update(self,):
        raise NotImplementedError  

    def metrics(self,x,y):
        raise NotImplementedError

    def train_step(self,x_in,y_truth): 
        '''
            returning y_pred so that a redundant call to forward isn't 
            required 
        '''
        ## need to properly define train_step (forward, backward, + update)
        y_layers = self.full_forward(x_in) #list of layers 
        grad = self.backward(y_layers, y_truth)
        self.update(grad)
        #do not absolutely need to return anything but it'll be convenient to have 
        # the model eval returned 
        return y_layers[-1]

    def fit(self,x_data,y_data,x_eval,y_eval, printing = False):
        ### This method implements our "1. forward 2. backward 3. update paradigm"
        ## it should call forward(), grad(), and update(), in that order. 
        # you should also call metrics so that you may print progress during training
        for epoch in range(self.epochs):
            y_pred = self.train_step(x_data,y_data)
            if printing and (epoch % 25 == 0):
                m_tr= self.metrics(y_pred,y_data)
                y_pred_eval = self.forward(x_eval)
                m_te = self.metrics(y_pred_eval,y_eval)
                a_disc = np.abs(m_te[1] - m_tr[1])
                l_disc = np.abs(m_te[0] - m_tr[0])
                print(f"epoch {epoch}: acc {m_te[1]:.3f}, acc discrep {a_disc:.3f}, loss disc {l_disc:.3f}")


class FCNetFS(MLPipeline):
    def __init__(self,params:list=None, dims:list=None,epochs:int = 10,lr = .01,):
        super().__init__(epochs = epochs,lr = lr)
        if params is None:
            weights = []
            bias = []
            for j in range(int(len(dims)-1)):
                w = (np.random.random([dims[j+1],dims[j]])-.5)
                b = (np.random.random(dims[j+1]) - .5)
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
        self.lr = lr 

    def y_pred(self,input,inIsX = False,thresh = .5):
        if inIsX:
            y_score = self.forward(input)
        else:
            y_score = input 
        y_out = 0 * y_score
        y_out[y_score >= thresh] = 1
        return y_out.astype(int)
    
    def forward(self,x):
        la = self.full_forward(x)
        a = la[-1]
        return a
   
    def full_forward(self,x):
        j = 0
        a = x 
        layer_acts = [a]
        for weight in self.weights:
            z = weight @  a + self.bias[j].reshape([self.bias[j].shape[0],1])
            a = self.activation(z)
            layer_acts.append(a)
            j+=1
        return layer_acts

    def backward(self,y_layers,y_truth):
        ''' your work is here!:
        y_layers = [a^0=x,a^1,...,a^L+1 = nu(x)] 
        '''
        y_score = y_layers[-1] # grab the output of network
        _,dc = self.cost(y_score,y_truth)#start with gradient of cost wrt output (should be 2 x m) 
        # we initialize lists to track gradients. 
        # note that parameters w and b are held in lists in the class 
        # w = [w^0,w^1,\ldots] and similarly for b, corresponding to [z^0,z^1,...]
        db_list = []
        dw_list = []
        for j in range(len(y_layers)-1):
            ''' recalling that backprop is "just" the chain rule + keep track of some info 
            at each step, we're going to layer-wise compute 
            dc/dparam = dc/dnu * dnu / dz^j * dz^j / dparam.
            We saw in class that we'll basically have a product of dz^{j+1}/dz^j
            which is dz^{j+1} / da^{j+1} * da^{j+1} / dz^j. While this is true, 
            for the first step, we'll only use one of these, and therefore split the product
            by computing the other half at the end of the loop
            '''

            idx = -(j+1) 
            y_layer = y_layers[idx] #previous activation function
            dadz = y_layer * (1-y_layer) #this is basically sigma'(z)
            dc *=  dadz ## we are going to use dc to track the backprop,
            ## i.e. dc is a glorified history of our use of the chain rule  
            a_prev = y_layers[idx-1] ## we need the previous layer's activation for dc/dw
            db = dc.mean(axis = 1) #I'll give you db 
            dw = (a_prev @ dc.transpose()).transpose()/a_prev.shape[1]# make sure you get the dimensions aligned. There may be some 
            #transposing required. Use of the debugger will come in handy here. 
            db_list.append(db)
            dw_list.append(dw)
            weight = self.weights[idx]
            dc =  weight.transpose() @ dc ## Now you can use the other half of the dz^j+1/dz^j computation 
            ## recall that we are mapping R^{n_{j+1}} --> R^{n_j}!, so you might have 
            # or need another judicious use of transpose 
        # reversing the list because in gradient update we'll iterate through params, 
        # and they're stored starting with early layers first
        db_list.reverse()
        dw_list.reverse()
        grad = zip(dw_list,db_list)
        return grad

    def update(self,grad):
        j = 0 
        for dw,db in grad:
            self.weights[j]  = self.weights[j] - self.lr * dw 
            self.bias[j] = self.bias[j] - self.lr * db 
            j += 1
        return None


    def metrics(self,x,y):
        return None

    def activation(self,z):
        return self.sigmoid(z)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def cost(self,y_score,y_truth):
        # your cost function should return cross entropy cost (sum over classes)
        c = -y_truth*np.log(y_score) - (1-y_truth)*np.log(1-y_score) #cross entropy
        dcdys  = -y_truth/y_score + (1-y_truth)/(1-y_score) #gradient of cost wrt to output
        return c.sum(axis = 0),dcdys #SUMMATION IS VERTICAL
    
    def metrics(self,y_score,y_truth):
        y_pred = self.y_pred(y_score)
        loss = self.loss(y_score,y_truth)
        acc = (y_pred == y_truth).mean()
        return loss, acc
    
    def loss(self,y_sc,y_truth):
        ### calculating loss using partial as initial computation
        c,_ = self.cost(y_sc,y_truth)
        return c.mean()

    def l_grad(self,x_in,y_truth):
        ### partial loss / partial y_pred 
        return None


if __name__ == "__main__":
    # Step 0: generate data and look at class separation 
    input_dim = 1
    dg = DataGen(dim = input_dim)
    ###a larger mu_factor will better separate means 
    ## and therefore class data 
    # We'll keep this large before getting backprop running
    xtr,ytr,xte,yte = dg.gen_data(mu_factor = 5)
    dg.probe_dat(xtr,ytr)
    '''
    # plt.show()
    # plt.clf()
    '''
   # Step 1: instantiate 1 hidden layer nn for binary classification task 
    # and define cost function returning cost + d cost/d nu 
    anet = FCNetFS(dims = [input_dim,20,2],epochs = 150,lr = 1)
    #sanity check that dcdnu is pointed in the right direction! 
    y_score = anet.forward(xtr)
    c,dc = anet.cost(y_score,ytr)
    ### figure out whether the following is + or - 
    ##... and don't be lazy, don't *just* try each and see what
    # catches the following 'if' statement 
    dc_dir = 0 
    dc_dir  = (np.sign(dc) == np.sign(ytr - y_score)).mean()
    if dc_dir == 1:
        print('dc/dnu points in the right direction!')

    #### Step 2: define backward() method in FCNetFS and
    ### train_step() (forward, backward, update) in MLPipeline
    ## if probe_dat() shows a plot with separated classes, your 
    # accuracy should hit 100%
    anet.fit(xtr,ytr,xte,yte,printing = True)
 
    ### Step 3: let's make things harder, just check that you can 
    ## still construct a good model
    input_dim = 15
    dg = DataGen(dim = input_dim)
    xtr,ytr,xte,yte = dg.gen_data(mu_factor = .5)
    # you may not see much now separation with probe_dat()
    
    for j in range(input_dim):
            plt.figure(j)
            dg.probe_dat(xtr,ytr,j)
    plt.show()
    bnet = FCNetFS(dims = [input_dim,20,15,10,2],epochs = 500,lr = 1)
    bnet.fit(xtr,ytr,xte,yte,printing = True)
