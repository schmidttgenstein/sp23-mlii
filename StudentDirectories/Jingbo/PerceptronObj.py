
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

class DataGen:
    def __init__(self, dim = 10):
        w = np.random.random(dim) - 0.5 # 1 by 10 vector; each entry [-0.5,0.5]
        w = w/la.norm(w) #normalize weight 
        self.w = w
        self.dim = dim

    def gen_data(self, n_data = 10000):
        x_data = 25 * (np.random.random([self.dim, n_data]) - 0.5) #10 by 10000 matrix
        y_data = np.sign(self.w @ x_data) #(-1,0,1)
        return x_data, y_data

    def filter_data(self, x_data, y_data): 
        #filtering out entries with y = 0 and eliminate data that are too close to 
        #the seperation hyperplane. In order to run this example faster
        g_idx = (y_data * (self.w @ x_data)) >= 1 #Perceptron -1 if wx < 0, 1 otherwise
        return x_data[:, g_idx], y_data[g_idx]

    def gen_marg_data(self, n_data = 10000):
        x,y = self.gen_data(n_data = n_data)
        xf, yf = self.filter_data(x,y)
        return xf, yf

class Perceptron:
    def __init__(self, dim = 10):
        w = np.random.random(dim) - 0.5
        w = w/la.norm(w)
        self.w = w
        self.dim = dim

    def get_wrong(self, x_data, y_data): #find wrongly labeled data
        x_w = self.w @ x_data
        x_y = x_w * y_data
        wrong_idx = x_y < 0
        b_num = sum(wrong_idx)
        x_wrong = x_data[:, wrong_idx]
        y_wrong = y_data[wrong_idx]
        return b_num, x_wrong, y_wrong

    def fit_w(self, x_data, y_data, printing = False):
        n, x_w, y_w = self.get_wrong(x_data, y_data)
        j = 0

        while n > 0: # perceptron goal: no more misclassified data
            rand_idx = np.random.randint(n) # randomly pick an integer < n 
            w = self.w + (y_w[rand_idx] * x_w[:, rand_idx] )
            self.w = w
            n, x_w, y_w = self.get_wrong(x_data, y_data) #check wrongly labeled data with updated w
            j = j + 1

            if printing:
                if j%25 == 0:
                    print('iteration: %d, wrong numbers: %d' % (j, n))

        bound = (la.norm(self.w) * la.norm(x_data,axis = 0).max())**2 
        return j, bound

    def eval_w (self, x_test, y_test):
        n, x, y = self.get_wrong(x_test, y_test)

        return 1.0 - float(n/x_test.shape[1]) #accuracy


def plot_2D(n = 1000,show = True):
    plt.clf()
    dg2 = DataGen(dim = 2)
    x,y = dg2.gen_data(n_data = n)
    xf,yf = dg2.filter_data(x,y)
    p_idx = y > 0 
    p_idx_fil = yf > 0
    plt.plot(x[0,p_idx],x[1,p_idx],'.',label = 'y>0')
    plt.plot(x[0,~p_idx],x[1,~p_idx],'.',label = 'y<0')
    plt.plot(xf[0,p_idx_fil],xf[1,p_idx_fil],'.',label = 'y>0 margin')
    plt.plot(xf[0,~p_idx_fil],xf[1,~p_idx_fil],'.',label = 'y<0 margin')
    plt.legend()
    if show:
        plt.show()

def one_run(n = 10000, dim =10):
    dg = DataGen(dim = dim)
    x,y = dg.gen_marg_data(n_data = n)
    x_test,y_test = dg.gen_data(n_data = n)
    perceptron = Perceptron(dim = dim)
    pre_tr_acc = perceptron.eval_w(x_test,y_test)
    perceptron.fit_w(x,y)
    post_tr_acc = perceptron.eval_w(x_test,y_test)
    print(f"pre and post train accuracies: {pre_tr_acc:.3f} and {post_tr_acc:.3f}")

def many_runs(n =50000, dims =np.linspace(2,30,15).astype(int)):
    data = np.zeros([4,dims.shape[0]])
    j = 0 
    for dim in dims:
        dg = DataGen(dim = dim)
        x,y = dg.gen_marg_data(n_data = n)
        assert x.shape[1]>0, "not enough data!"
        xt,yt = dg.gen_data(n_data = n)
        perceptron = Perceptron(dim = dim)
        n_steps,upper_bound = perceptron.fit_w(x,y,printing = False) 
        accuracy = perceptron.eval_w(xt,yt)
        data[:,j] = np.array([dim,n_steps,upper_bound,accuracy])
        j+=1
    plt.figure(1)
    plt.plot(data[0,:],data[3,:],'-.')
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.figure(2)
    plt.plot(data[0,:],data[1,:],'-.',label = 'steps to convergence')
    plt.legend()
    plt.xlabel('dimension')
    plt.ylabel('number of steps')
    plt.show()

if __name__ == "__main__":
    ## Step 1: visualize data
    plot_2D(n = 10000)
    ## Step 2: Check instance performance of model 
    one_run(n= 1000)
    ## Step 3: Check complexity, empirically
    many_runs()

            



    




        
