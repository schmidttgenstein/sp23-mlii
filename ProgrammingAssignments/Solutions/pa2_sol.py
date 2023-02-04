import numpy as np 
import pandas as pd 
import numpy.linalg as la 
import numpy.matlib
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')

### You  will not use MLPipeline in this assignment;
## we're including it for reference because you will be using 
# it soon enough, and also another class inherits from it 
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


class UniversalApprox:
    def __init__(self,epsilon=0.01,fun = None):
        self.eps = epsilon
        if fun == None:
            self.fun = self.wiggles_fun
        else:
            self.fun = fun 

    def sigma(self,t,a=1,b=0,):
        s = 1/(1+np.exp(-(t-b)/a))
        return s 
    
    def solve_ab(self,x,y):
        ## solve for a,b in sigma
        l0 = y[0]/(1-y[0])
        l1 = y[1]/(1-y[1])
        A = np.array([[np.log(l0), 1],[np.log(l1),1]])
        b = np.array([[x[0]],[x[1]]])
        sol = np.linalg.inv(A) @ b 
        return sol
    
    def find_nn_params(self,t_in,fun,j_ext =10):
        t_ext = np.linspace(t_in[0]-t_in[j_ext],t_in[-1]+(t_in[j_ext]-t_in[0]),t_in.shape[0]+int(2*j_ext))
        delta = t_in[1] - t_in[0]
        n = t_ext.shape[0]
        alphabet = np.zeros([2,n])
        for j,t in enumerate(t_ext):
            x_pair = np.array([t_ext[j],t_ext[j]-delta/2])
            y_pair = np.array([1-self.eps/2,.5])
            ab = self.solve_ab(x_pair,y_pair)
            alphabet[:,j] = ab.flatten()
        samples = fun(t_ext)
        return t_ext, samples, alphabet

    def approx_fun(self,t_in,fun,j_ext = 10,t_eval = None):
        ### We compute a po1-approx approximation of function fun 
        ## the first step is to slightly extend the domain so that po1~ 
        # approximates at the tails. This is just an annoying accounting detail
        t_ext, samples, alphabet = self.find_nn_params(t_in,fun,j_ext)
        if t_eval is not None:
            t_ext = t_eval
        y = np.zeros(t_ext.shape[0])
        y_tot = np.zeros([t_ext.shape[0],samples.shape[0]])
        n = samples.shape[0]
        for j,_ in enumerate(range(samples.shape[0])):
            if j < n - 1:
                rho = samples[j]*(self.sigma(t_ext,alphabet[0,j],alphabet[1,j]) - self.sigma(t_ext,alphabet[0,j+1],alphabet[1,j+1]))
                y_tot[:,j] = rho
                y = y + rho
        return t_ext, y, y_tot

    def wiggles_fun(self,t):
        return np.sin(5*t)*np.exp(t)


class UANet(MLPipeline):
    def __init__(self,params):
        super().__init__()
        w_temp =np.matlib.repmat(1/params[0],1,2)
        b_temp = np.matlib.repmat(params[1],2,1)
        f_temp = np.matlib.repmat(params[2],2,1)
        f_temp[1,:] = -1 * f_temp[1,:]
        F_temp = f_temp
        bt = b_temp.reshape((b_temp.shape[1],2),order = 'F').reshape((1,b_temp.shape[1]*2),order = 'F')
        ft = F_temp.reshape((F_temp.shape[1],2),order = 'F').reshape((1,F_temp.shape[1]*2),order = 'F')
        bias = bt[:,1:-1]
        weights = w_temp[:,1:-1]
        self.weights = weights 
        self.bias = bias 
        self.f_val = ft[:,:-2]

    def forward(self,x):
        la = self.full_forward(x)
        a = la[-1]
        return a

    def full_forward(self,x):
        z = self.weights.transpose() @ x - self.weights.transpose() * nn.bias.transpose()
        a = self.sigmoid(z)
        ff = self.f_val @ a
        return  ff

    def backward(self,x_in,y_truth):
        return None 

    def update(self,grad):
        return None 

    def metrics(self,x,y):
        return self.loss(x,y)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def loss(self,x_in,y_truth):
        ### calculating loss using partial as initial computation
        l_grad, _ = self.l_grad(x_in,y_truth)
        return (.25 * l_grad**2).mean()

    def l_grad(self,x_in,y_truth):
        return None




if __name__ == "__main__":
    plotting = True
    eps = .01
    ## Step 1: instantiate your UniversalApprox Object
    ua = UniversalApprox(epsilon = eps)


    
    ## Step 2: Generate approximations with po1
    t_space = np.linspace(0,3,60)
    t_ext = np.linspace(0,3,2500)

    x_points = np.array([0,2])
    y_points = np.array([eps,1-eps])
    ab = ua.solve_ab(x_points,y_points)
    sigm = ua.sigma(t_ext,ab[0],ab[1])
    plt.figure(0)
    plt.plot(t_ext,sigm)

    t_eval, y_approx,y_incs  = ua.approx_fun(t_space,ua.wiggles_fun,t_eval = t_ext)
    y_truth = ua.fun(t_eval)
    y_err = y_truth - y_approx
    if plotting: 
        plt.figure(1)
        plt.plot(t_eval,y_approx,label = 'approximation w/ po1')
        plt.plot(t_eval,y_truth, label = 'truth')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('y')
        plt.figure(2)
        plt.plot(t_eval,y_err)
        plt.xlabel('t')
        plt.ylabel('err')
        plt.figure(3)
        for j in range(y_incs.shape[1]):
            plt.plot(t_eval,y_incs[:,j])
        plt.xlabel('t')
        plt.ylabel('y')


    ## Step 3: Extract weights for Neural Net and instantiate object
    _,s,ab = ua.find_nn_params(t_in = np.linspace(0,3,250),fun = ua.wiggles_fun,)
    params = [ab[0],ab[1],s]
    nn = UANet(params)

    ## Step 4: Check nn approximation
    t_space = np.linspace(0,4,10000)
    y_nn = nn.forward(t_space.reshape((1,t_space.shape[0])))
    y_truth = ua.fun(t_space)
    err = y_nn - y_truth 
    if plotting:
        plt.figure(4)
        plt.plot(t_space,y_nn,label = 'nn output')
        plt.plot(t_space,y_truth,label = 'truth')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend()
        plt.figure(5)
        plt.plot(t_space,err)
        plt.xlabel('t')
        plt.ylabel('err')
        plt.show()

   