
import time
import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import numpy.linalg as la 
import torch.optim as optim
import matplotlib.pyplot as plt 
from collections import OrderedDict


class DataGenerator:
    def __init__(self,dim = 10,N = 10000):
        self.dim = dim 
        self.N = N 

    def gen_data(self,n = None,split = 0.6,mu_factor = 1):
        if n is None:
            n = self.N 
        x0,y0 = self.gen_label_data(label = 0,n_dat = n,mult = mu_factor)
        x1,y1 = self.gen_label_data(label = 1,n_dat = n,mult = mu_factor)
        cutoff = int(split * x0.shape[1])
        idx = np.arange(n)
        np.random.shuffle(idx)
        x_tr = np.concatenate([x0[:,:cutoff],x1[:,:cutoff]],axis = 1)
        y_tr = np.concatenate([y0[:,:cutoff],y1[:,:cutoff]],axis = 1)
        x_te = np.concatenate([x0[:,cutoff:],x1[:,cutoff:]],axis = 1)
        y_te = np.concatenate([y0[:,cutoff:],y1[:,cutoff:]],axis = 1)
        x_tr = x_tr[:,idx]
        y_tr = y_tr[:,idx]
        np.random.shuffle(idx)
        x_te = x_te[:,idx]
        y_te = y_te[:,idx]
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

    def probe_dat(self,x,y,coord = 0,show = False):
        yo = self.ym2yo(y)
        idx1 = yo == 1 
        x1 = x[:,idx1]
        x0 = x[:,~idx1]
        if show:
            plt.hist(x0[coord,:],density = True, bins = 100,label = 'x(y=0)')
            plt.hist(x1[coord,:],density = True, bins = 100,label = 'x(y=1)')
            plt.legend()
            plt.show()

class DataSet:
    def __init__(self,x,y, tor = False):
        if tor:
            self.x = torch.tensor(x).float() 
            self.y = torch.tensor(y).int()
        else:
            self.x = x 
            self.y = y
        self.n_samples = x.shape[1]

    def __len__(self):
        return self.n_samples 
    
    def __getitem__(self,idx):
        return self.x[:,idx], self.y[:,idx]

class DataBatcher:
    def __init__(self,dataset,batch_size = 1):
        self.length = dataset.__len__() 
        self.batch_size = batch_size
        self.counter = 0 
        self.dataset = dataset

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.counter * self.batch_size >= self.length:
            raise StopIteration 
        else:
            c_idx = self.counter 
            bs = self.batch_size 
            start_idx = c_idx * bs 
            end_idx = np.min([(c_idx+1)*bs,self.length])
            self.counter = c_idx + 1
            idxs = np.arange(start_idx,end_idx)
            return self.dataset[idxs]

class DataLoader:
    def __init__(self,dataset,batch_size = 1,tor = False):
        if tor:
            self.dataset = dataset
        else:
            self.dataset = dataset 
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(dataset.__len__() / batch_size))

    def __iter__(self):
        return DataBatcher(self.dataset,self.batch_size)







class MLPipeline(nn.Module):
    def __init__(self,epochs = 10,lr = 0.025):
        super().__init__()
        ###In this constructor we set the model complexity, number of epochs for training, 
        ##and learning rate lr. You should think of complexity here as "number of parameters"
        #defining model. In linear regression, this e.g. may be (deg of poly)-1. 
        self.epochs = epochs
        self.lr = lr 

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

    def train_step(self,x_in,y_truth):
        '''
            returning y_pred so that a redundant call to forward isn't 
            required 
        '''
        y_score = self.forward(x_in)
        grad = self.backward(y_score,y_truth)
        self.update(grad)
        return y_score

    def fit(self,train_loader,val_loader = None, printing = False):
        ### This method implements our "1. forward 2. backward 3. update paradigm"
        ## it should call forward(), grad(), and update(), in that order. 
        # you should also call metrics so that you may print progress during training
        p_mets = np.zeros([self.epochs,4])
        for epoch in range(self.epochs):
            trm_array = np.zeros([train_loader.num_batches,2])
            vam_array = np.zeros([val_loader.num_batches,2])
            for batch_idx,(x_data,y_data) in enumerate(train_loader):
                y_score = self.train_step(x_data,y_data)
                trm_array[batch_idx,:] = self.metrics(y_score,y_data)
            for batch_idx,(x_data,y_data) in enumerate(val_loader):
                y_score = self.forward(x_data)
                vam_array[batch_idx,:] = self.metrics(y_score,y_data)
            m_diff = vam_array.mean(axis = 0) - trm_array.mean(axis = 0)
            p_mets[epoch,:] = np.array([m_diff,vam_array.mean(axis =0)]).flatten()
            if epoch % 1 == 0:
                print(f"epoch {epoch}, loss_diff {p_mets[epoch,0]:.3f}, acc_diff {p_mets[epoch,1]:.3f}, loss {p_mets[epoch,2]:.3f}, accuracy {p_mets[epoch,3]:.3f}")
        return p_mets      

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


    def train_step(self,x_in,y_truth):
        '''
            overwriting train_step from MLPipeline since we use full forward instead of forward
            Notice that our torch version uses MLPipeline's train_step
        '''
        y_layers = self.full_forward(x_in)
        grad = self.backward(y_layers,y_truth)
        self.update(grad)
        return y_layers[-1]
    
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
        y_score = y_layers[-1]
        _,dc = self.cost(y_score,y_truth) 
        db_list = []
        dw_list = []
        for j in range(len(y_layers)-1):
            idx = -(j+1)
            y_layer = y_layers[idx]
            dadz = y_layer * (1-y_layer)
            dc *= dadz 
            a_prev = y_layers[idx-1]
            db = dc.mean(axis = 1)
            dw = (a_prev @ dc.transpose()).transpose()/a_prev.shape[1]
            db_list.append(db)
            dw_list.append(dw)
            weight = self.weights[idx]
            dc = weight.transpose() @ dc
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

    def activation(self,z):
        return self.sigmoid(z)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def cost(self,y_score,y_truth):
        # your cost function should return cross entropy cost (sum over classes)
        c = -y_truth*np.log(y_score) -(1-y_truth)*np.log(1-y_score)
        dcdys  = -y_truth/y_score + (1-y_truth)/(1-y_score)
        return c.sum(axis = 0),dcdys
    
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
        self.opt = optim.SGD(self.parameters(),lr = lr)
        self.loss_fun = nn.CrossEntropyLoss()

    def y_pred(self,input,inIsX = False,thresh = .5):
        if inIsX:
            y_score = self.forward(input)
        else:
            y_score = input 
        y_out = 0 * y_score
        y_out[y_score >= thresh] = 1
        return torch.transpose(y_out.int(),0,1)

    def metrics(self,y_score,y_truth):
        with torch.no_grad():
            y_pred = self.y_pred(y_score)
            loss = self.loss_fun(y_score,torch.transpose(y_truth.float(),0,1))
            acc = (y_pred == y_truth).float().mean()
        return loss, acc
    
    def forward(self,x_in):
        return self.forward_stack(torch.transpose(x_in,0,1))
    
    def backward(self,y_score,y_truth):
        self.opt.zero_grad()
        loss = self.loss_fun(y_score,torch.transpose(y_truth.float(),0,1))
        loss.backward()

    def update(self,grad = None ):
        self.opt.step()
        

if __name__ == "__main__":
    # Step 0: generate data and remind yourself what this data looks like
    input_dim = 2
    dg = DataGenerator(dim = input_dim,N = 100)
    x,y,_,_ = dg.gen_data(mu_factor = 2,split=.5)

    # Step 1: Make dataset, databatcher (iterator), and dataloader (iterable)
    # and sanity check what you get from each
    data_set = DataSet(x,y)
    for j in range(3):
        ## Check what getitem returns 
        print(data_set.__getitem__(j))
    print('-----------')
    for j in range(3):
        # another way to call getitem 
        print(data_set[j])

    # define the iterable
    data_loader = DataLoader(data_set,batch_size = 30)
    ## then instantiate the iterator
    data_batcher = iter(data_loader)
    j = 0 
    for batch in data_batcher:
        print(f"batch {j}, and y data {batch[1]} with {batch[1].shape[1]} samples")
        j+=1 
        # Notice how many samples are in the last batch. Why is it different?
    ## for kicks and giggles try running through iterator again
    k = j 
    for batch in data_batcher:
        print(f"batch {k}, and  {batch[1]} with {batch[1].shape[1]} samples")
        k+=1
    if k == j:
        print('exhausted data in iterator!')

    #Iterables, however, can be iterated with raw for loop, and you can repeat iteration
    j = 0 
    for batch in data_loader:
        print(f"batch {j} having {batch[1].shape[1]} samples")
        j+=1 
    k = j
    for batch in data_loader:
        print(f"batch {k} having {batch[1].shape[1]} samples")
        k+=1 
    if k != j:
        print('can reiterate through iterable in a for loop!')
    
    
    #Step 2: Run SGD using code from pa4 with FCNetFS object
    input_dim = 20
    eta = 0.5
    num_epochs = 25
    dg = DataGenerator(dim = input_dim,N = 2000000)
    xtr,ytr,xval,yval = dg.gen_data(mu_factor = .01,split=.5)
    ## Notice our mu factor, it's tiny, so component-wise the data is hardly separated
    ds_tr = DataSet(xtr,ytr)
    ds_val = DataSet(xval,yval)
    dl_tr = DataLoader(ds_tr,5000)
    dl_val = DataLoader(ds_val, 5000)
    anet = FCNetFS(dims = [input_dim,15,5,2],epochs = num_epochs,lr = eta)
    t0 = time.time()
    anet.fit(dl_tr,dl_val,printing = True)
    t1 = time.time()-t0 
    print('-------------')
    print(f"took {t1:3f} seconds to run {num_epochs} epochs")
    print('-------------')


    # Step 3: Run Torch Version of step 2
    # Note that data needs to be tensorified 
    ds_tr_to = DataSet(xtr,ytr,tor = True)
    ds_val_to = DataSet(xval,yval,tor = True)
    dl_tr_to = DataLoader(ds_tr_to,5000)
    dl_val_to = DataLoader(ds_val_to, 5000)
    cnet = FCNetTo(dims = [input_dim,15,5,2], epoch = num_epochs, lr=eta)
    t0 = time.time()
    cnet.fit(dl_tr_to,dl_val_to,printing = True)
    t1 = time.time()-t0 
    print('-------------')
    print(f"took {t1:3f} seconds to run {num_epochs} epochs")
    print('-------------')  


    ## Step 4: Hypothesize why, despite component-wise similarity in data
    ## these networks are able to well-discriminate classes