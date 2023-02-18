import numpy as np
import numpy.linalg as la
import numpy.matlib
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, epochs = 250, lr = 0.025):
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
        l0 = y[0]/(1-y[0])
        l1 = y[1]/(1-y[1])
        A = np.array([[np.log(l0), 1],[np.log(l1),1]])
        b = np.array([[x[0]],[x[1]]])
        sol = np.linalg.inv(A) @ b 
        return sol
    
    def find_nn_params(self,t_in,fun,j_ext =10):
        ext = np.linspace(t_in[0] - t_in[j_ext], t_in[-1] + (t_in[j_ext] - t_in[0]), t_in.shape[0] + int(2 * j_ext))
        samples = fun(ext)
        delta = t_in[1] - t_in[0]

        alphabet = np.zeros([2, ext.shape[0]])
        for idx in range(ext.shape[0]):
            x = np.array([ext[idx], ext[idx] - delta / 2])
            y = np.array([1 - self.eps / 2, .5])
            ab = self.solve_ab(x, y)
            alphabet[:, idx] = ab.flatten()

        return ext, samples, alphabet

    def approx_fun(self, t_in, fun, j_ext = 10, t_eval = None):

        ext, samples, alphabet = self.find_nn_params(t_in,fun,j_ext)

        if t_eval is not None:
            t_ext = t_eval

        ne = ext.shape[0]
        ns = samples.shape[0]
        y = np.zeros(ne)
        y_tot = np.zeros([ne, ns])

        for idx in range(ns-1):

            rho1 = samples[idx]*(self.sigma(ext,alphabet[0,idx],alphabet[1,idx]))
            rho2 = self.sigma(ext,alphabet[0,idx+1],alphabet[1,idx+1])
            rho = rho1 - rho2

            y_tot[:, idx] = rho
            y = y + rho

        return ext, y, y_tot

    def wiggles_fun(self,t):
        return np.sin(5*t)*np.exp(t)


class UANet(MLPipeline):
    def __init__(self,params):
        super().__init__()

        w_temp = 1/params[0]
        b_temp = params[1]
        f_temp = params[2]

        _w_temp = np.hstack((w_temp, w_temp)).reshape(1, 540)
        _b_temp = np.vstack((b_temp, b_temp))
        _f_temp = np.vstack((f_temp, f_temp))
        _f_temp[1, :] = -1 * _f_temp[1, :]


        bt = _b_temp.reshape((_b_temp.shape[1], 2), order='F').reshape((1, _b_temp.shape[1] * 2), order='F')
        ft = _f_temp.reshape((_f_temp.shape[1], 2), order='F').reshape((1, _f_temp.shape[1] * 2), order='F')

        self.weights = _w_temp[:, 1:-1]
        self.bias = bt[:, 1:-1]
        self.f_val = ft[:, :-2]


    def forward(self,x):
        z = self.weights.transpose() @ x - self.weights.transpose() * nn.bias.transpose()
        a = self.sigmoid(z)
        f = self.f_val @ a
        return f[-1]

    def backward(self,x_in,y_truth):
        ## Do not implement
        return None 

    def update(self,grad):
        ## Do not implement
        return None 

    def metrics(self,x,y):
        ## Do not (need to) implement
        return None

    def sigmoid(self,z):
        ## Already implemented for you
        return 1/(1+np.exp(-z))
    
    def loss(self,x_in,y_truth):
        ### calculating loss using partial as initial computation
        ## Do not implement
        return None

    def l_grad(self,x_in,y_truth):
        ## Do not implement
        return None




if __name__ == "__main__":

    plotting = True

    ## Step 1: instantiate your UniversalApprox Object
    eps = .01
    ua = UniversalApprox(epsilon = eps)
    x_lim = 3

    ### Step 2: Generate approximations with po1
    ## If you use my wiggles_fun, it has an exp, so don't go wild on your domain
    # need sample linspace for function approximation, and denser linspace for evaluation
    t_space = np.linspace(0,x_lim,30)
    t_dense = np.linspace(0,x_lim,2500)
    
    ###2.a: check the a,b solve
    ## expect y values of sigmoidal to correspond to those provided (y_points)
    # at specified x values (x_points). Play around with this until you understand it
    x_points = np.array([0,3])
    y_points = np.array([eps,1-eps])
    ab = ua.solve_ab(x_points,y_points)
    sigm = ua.sigma(t_dense,ab[0],ab[1])
    plt.figure(0)
    plt.plot(t_dense,sigm)

    ### 2.b, th bulk of your work will be here: define the approximation method using 
    ## c-inf bump functions, as defined by successive differences of sigmoidals
    t_eval, y_approx,y_incs  = ua.approx_fun(t_space,ua.wiggles_fun,t_eval = t_dense)
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
    _,s,ab = ua.find_nn_params(t_in = np.linspace(0,x_lim,250),fun = ua.wiggles_fun,)
    params = [ab[0],ab[1],s]
    nn = UANet(params)

    ### Step 4: Check nn approximation
    ## extending domain to show that the nn will only work for the domain of approximations 
    # you found earlier
    t_space = np.linspace(-1,x_lim+1,10000)
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

   