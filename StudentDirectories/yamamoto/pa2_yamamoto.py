import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# You  will not use MLPipeline in this assignment;
# we're including it for reference because you will be using
# it soon enough, and also another class inherits from it 
class MLPipeline:
    def __init__(self, epochs=250, lr=0.025):
        # In this constructor we set the model complexity, number of epochs for training,
        # and learning rate lr. You should think of complexity here as "number of parameters"
        # defining model. In linear regression, this e.g. may be (deg of poly)-1.
        self.epochs = epochs
        self.lr = lr

    def gen_data(self, ):
        raise NotImplementedError

    def loss(self, ):
        raise NotImplementedError

    def forward(self, ):
        raise NotImplementedError

    def backward(self, ):
        raise NotImplementedError

    def update(self, ):
        raise NotImplementedError

    def metrics(self, x, y):
        raise NotImplementedError

    def fit(self, x_data, y_data, x_eval, y_eval, printing=False):
        # This method implements our "1. forward 2. backward 3. update paradigm"
        # it should call forward(), grad(), and update(), in that order.
        # you should also call metrics so that you may print progress during training
        if printing:
            self.x_eval = x_eval
            self.y_eval = y_eval
        for epoch in range(self.epochs):
            y_pred = self.forward(x_data)
            loss = self.loss(y_pred, y_data)
            grad = self.backward(x_data, y_data)
            self.update(grad)
            if printing:
                m = self.metrics(x_eval, y_eval)
                if epoch % 100 == 0:
                    print(f"epoch {epoch} and train loss {loss.mean():.2f}, test metrics {m:.2f}")
        if printing:
            self.m = m


class UniversalApprox:
    def __init__(self, epsilon=0.01, fun=None):
        self.eps = epsilon
        if fun is None:
            # you are welcome to define your own custom function
            self.fun = self.wiggles_fun
        else:
            self.fun = fun

    def sigma(self, t, a=1, b=0):
        """
            This way of expressing sigmoidal is to facilitate
            solve_ab, to make transparent 'linearity' wrt unknowns
            You may not even need to use solve_ab, in the event you
            work out simpler math yourself (cf class)
        """
        s = 1 / (1 + np.exp(-(t - b) / a))
        return s

    def solve_ab(self, x, y):
        # solves for a,b in sigma above
        # ensure you understand how / what this solves for
        # x[0] = y[0] and x[1] = y[1]
        l0 = y[0] / (1 - y[0])
        l1 = y[1] / (1 - y[1])
        A = np.array([[np.log(l0), 1], [np.log(l1), 1]])
        b = np.array([[x[0]], [x[1]]])
        sol = np.linalg.inv(A) @ b
        return sol

    def find_nn_params(self, t_in, fun, j_ext=10):
        """
            This method finds the a and bs for sigmoidals which can be used
            for po1 approximate (bump functions in programming assignment)
            You should return
                1. your domain-extended input array (need to extend domain for po1 to cover)
                2. function evaluations at bump function's apex
                3. array of a,b parameters defining the bump functions
        """
        t_min = t_in[0]
        t_max = t_in[-1]
        n_extended = t_in.shape[0]+2*j_ext
        t_extended = np.linspace(t_min-t_in[j_ext], t_max+t_in[j_ext], n_extended)
        f_eval = fun(t_extended)
        ab_params = np.zeros([n_extended, 2])
        delta_t = t_in[1] - t_in[0]
        for j, t in enumerate(t_extended):
            x = np.array([t_extended[j], t_extended[j] - delta_t])
            y = np.array([1-self.eps/2, 0+self.eps/2])
            ab = self.solve_ab(x, y)
            # print("x:", x, "y:", y, "alpha:", ab[0], "beta:", ab[1])
            ab_params[j, :] = ab.reshape((1, 2))
        return t_extended, f_eval, ab_params

    def approx_fun(self, t_in, fun, j_ext=10, t_eval=None, plotting=False):
        """
            This method uses the bump function parameters you computed
            from find_nn_parameters, and returns a function's approximation as
            y(x) = sum_i y(x_i)bump(x) (where sum is over sampled values)
            returns
                1. t_space (filtered)
                2. y approximation evaluated on t_space
                3. individual y approximation bumps, as large array
                    (see figure from programming assignment)
            There's a subtle point with t_eval: you will compute approximation based on t_in,
            which will likely have courser granularity than t_eval. Because you may want something smooth looking,
            you'll use t_eval to plot *output*
        """
        if plotting:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 3)
            ax3 = fig.add_subplot(2, 2, 2)
            ax4 = fig.add_subplot(2, 2, 4)
            ax1.title.set_text("Sigmoids")
            ax2.title.set_text("Bumps")
            ax3.title.set_text("Scaled bumps")
            ax4.title.set_text("Approximation")
        t_space, f_eval, ab_params = self.find_nn_params(t_in, fun, j_ext)
        if t_eval is not None:
            t_space = t_eval
        n_bumps = t_space.shape[0]
        n_f = f_eval.shape[0]
        y_approx = np.zeros(n_bumps)
        yi_bumps = np.zeros([n_f-1, n_bumps])
        for j in np.arange(0, n_f-1):
            current_sigma = self.sigma(t_space, ab_params[j, 0], ab_params[j, 1])
            next_sigma = self.sigma(t_space, ab_params[j+1, 0], ab_params[j+1, 1])
            bump = current_sigma - next_sigma
            bump_scaled = f_eval[j] * bump
            yi_bumps[j, :] = bump_scaled
            y_approx += bump_scaled
            if plotting:
                ax1.plot(t_space, current_sigma)
                if j == n_f-1:
                    ax1.plot(t_space, next_sigma)
                ax2.plot(t_space, bump)
                ax3.plot(t_space, bump_scaled)
        if plotting:
            ax4.plot(t_space, self.fun(t_space), label="f(t)")
            ax4.plot(t_space, y_approx, label="approximation using po1")
            ax4.legend(loc="upper left")
            fig.tight_layout()
            plt.show()
        return t_space, y_approx, yi_bumps

    def wiggles_fun(self, t):
        return np.sin(5 * t) * np.exp(t)


class UANet(MLPipeline):
    def __init__(self, params):
        super().__init__()
        """
            Instantiation of this class will likely be your biggest headache. You need to make sure you 
            take parameters and align / shape them correctly for your network. Will need weights, biases, 
            and function sampled evaluations (which you get from UniversalApprox.find_nn_params); just make 
            sure that you process and store them correctly!
            Aside from hashing out the constructor, you will only need to implement the forward method for this 
            class, how simple!
        """
        w_temp = 1/params[0]  # TODO: why 1/w?
        w_temp_concat = np.concatenate((w_temp, w_temp), axis=0).reshape((1, w_temp.shape[0]*2))  # TODO: why duplicate w's?
        weights = w_temp_concat[:, 1:-1]  # TODO: how does this even make sense?
        b_temp = params[1]
        b_temp_stack = np.vstack((b_temp, b_temp))  # TODO: why duplicate b's?
        b_temp_reshape = b_temp_stack.reshape((1, b_temp_stack.shape[1]*2), order="F")  # TODO: why?
        bias = b_temp_reshape[:, 1:-1]  # TODO: how does this even make sense?
        f_temp = params[2]
        f_temp_stack = np.vstack((f_temp, f_temp))  # TODO: why duplicate f's?
        f_temp_stack[1, :] = -1 * f_temp_stack[1, :]  # TODO: why alternate f between + and -?
        f_temp_reshape = f_temp_stack.reshape((1, f_temp_stack.shape[1]*2), order="F")  # TODO: why?
        f_val = f_temp_reshape[:, :-2]  # TODO: why is this one different (i.e., [:, :-2] instead of [:, 1:-1])?
        self.weights = weights
        self.bias = bias
        self.f_val = f_val

    def forward(self, x):
        # computes the forward pass for network x -> z = wx+b -> a = sigm(z) -> c*a
        z = self.weights.transpose() @ x - self.weights.transpose() * self.bias.transpose()  # TODO: why multiply bias by weight?
        a = self.sigmoid(z)
        f = self.f_val @ a
        g = f[-1]
        return g

    def backward(self, x_in, y_truth):
        # Do not implement
        return None

    def update(self, grad):
        # Do not implement
        return None

    def metrics(self, x, y):
        # Do not (need to) implement
        return None

    def sigmoid(self, z):
        # Already implemented for you
        return 1 / (1 + np.exp(-z))

    def loss(self, x_in, y_truth):
        # calculating loss using partial as initial computation
        # Do not implement
        return None

    def l_grad(self, x_in, y_truth):
        # Do not implement
        return None


if __name__ == "__main__":
    plotting = True

    # Step 1: instantiate your UniversalApprox Object
    eps = .01
    ua = UniversalApprox(epsilon=eps)

    # Step 2: Generate approximations with po1
    # If you use my wiggles_fun, it has an exp, so don't go wild on your domain
    # need sample linspace for function approximation, and denser linspace for evaluation
    x_lim = 3
    t_space = np.linspace(0, x_lim, 60)
    t_dense = np.linspace(0, x_lim, 2500)
    # 2.a: check the a,b solve
    # expect y values of sigmoidal to correspond to those provided (y_points)
    # at specified x values (x_points). Play around with this until you understand it
    x_points = np.array([0, 2])
    y_points = np.array([eps, 1-eps])
    ab = ua.solve_ab(x_points, y_points)
    sigm = ua.sigma(t_dense, ab[0], ab[1])
    # if plotting:
    #     plt.figure(0)
    #     plt.plot(t_dense, sigm)
    #     plt.show()
    # 2.b, th bulk of your work will be here: define the approximation method using
    # c-inf bump functions, as defined by successive differences of sigmoidals
    t_eval, y_approx, y_incs = ua.approx_fun(t_space, ua.wiggles_fun, t_eval=t_dense, plotting=plotting)
    y_truth = ua.fun(t_eval)
    y_err = y_truth-y_approx
    # if plotting:
    #     plt.figure(1)
    #     plt.plot(t_eval, y_approx, label="approximation w/ po1")
    #     plt.plot(t_eval, y_truth, label="truth")
    #     plt.legend()
    #     plt.xlabel("t")
    #     plt.ylabel("y")
    #     plt.show()
    #     plt.figure(2)
    #     plt.plot(t_eval, y_err)
    #     plt.xlabel("t")
    #     plt.ylabel("err")
    #     plt.show()
    #     plt.figure(3)
    #     for j in range(y_incs.shape[0]):
    #         plt.plot(t_eval, y_incs[j, :])
    #     plt.xlabel("t")
    #     plt.ylabel("y")
    #     plt.show()

    # Step 3: Extract weights for Neural Net and instantiate object
    _, s, ab = ua.find_nn_params(t_in=np.linspace(0, x_lim, 250), fun=ua.wiggles_fun)
    params = [ab[:, 0], ab[:, 1], s]
    nn = UANet(params)

    # Step 4: Check nn approximation
    # extending domain to show that the nn will only work for the domain of approximations
    # you found earlier
    t_space = np.linspace(-1, x_lim+1, 10000)
    y_nn = nn.forward(t_space.reshape((1, t_space.shape[0])))
    y_truth = ua.fun(t_space)
    err = y_nn-y_truth
    if plotting:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.title.set_text("Model predictions vs ground truth")
        ax2.title.set_text("Error")
        ax1.plot(t_space, y_truth, label='truth')
        ax1.plot(t_space, y_nn, label='nn output')
        ax1.set(xlabel='t', ylabel='y')
        ax1.legend()
        ax2.plot(t_space, err)
        ax2.set(xlabel='t', ylabel='err')
        fig.tight_layout()
        plt.show()
