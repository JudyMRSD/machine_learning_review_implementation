# https://github.com/peterroelants/peterroelants.github.io/blob/master/notebooks/neural_net_implementation/neural_network_implementation_part01.ipynb
# Python imports
import numpy  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library

# Allow matplotlib to plot inside this notebook
# Set the seed of the numpy random number generator so that the tutorial is reproducable
numpy.random.seed(seed=1)

class basic_nn():
    def __init__(self):
        pass
    # Generate the target values t from x with small gaussian noise so the estimation won't
    # be perfect.
    # Define a function f that represents the line that generates t without noise

    def f(self, x):
        return x * 2

    def forward(self, x, w):
        return x*w

    def cost(self,y, t):
        # can also use mean, but with different learning rate
        return ((t-y)**2).sum()

    def gradient(self, w, x, t):
        return 2*x*(self.forward(x, w) - t)

    def delta_w(self, w, x, t, learning_rate):
        return learning_rate * self.gradient(w, x, t).sum()

    def back_prop(self, w, x, t, learning_rate):
        dw = self.delta_w(w, x, t, learning_rate)
        w = w - dw
        return w

class test_nn():
    def __init__(self):
        self.nn = basic_nn()

    def test_gradient_descent(self):
        w = 0.1
        learning_rate = 0.1
        # Start performing the gradient descent updates, and print the weights and cost:
        num_iters = 4  #number of gradient descent updates
        # List to store the weight,costs values, just for visualization
        # first row is the weight and cost before gradient descent
        w_cost = [(w, self.nn.cost(self.nn.forward(self.x, w), self.t))]
        for i in range(num_iters):
            w = self.nn.back_prop(w, self.x, self.t, learning_rate)
            # just for visualize , not part of gradient descent
            w_cost.append((w, self.nn.cost(self.nn.forward(self.x, w), self.t)))  # Add weight,cost to list
        # Print the final w, and cost
        for i in range(0, len(w_cost)):
            print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))

        # Plot the first 2 gradient descent updates
        plt.plot(self.ws, self.cost_ws, 'r-')  # Plot the error curve
        # Plot the updates
        for i in range(0, len(w_cost) - 2):
            w1, c1 = w_cost[i]
            w2, c2 = w_cost[i + 1]
            plt.plot(w1, c1, 'bo')
            plt.plot([w1, w2], [c1, c2], 'b-')
            plt.text(w1, c1 + 0.5, '$w({})$'.format(i))
            # Show figure
        plt.xlabel('$w$', fontsize=15)
        plt.ylabel('$\\xi$', fontsize=15)
        plt.title('Gradient descent updates plotted on cost function')
        plt.grid()
        plt.savefig("./result/gradient_descent.png")

    def define_data(self):
        # Define the vector of input samples as x, with 20 values sampled from a uniform distribution
        # between 0 and 1
        self.x = numpy.random.uniform(0, 1, 20)
        # Create the targets t with some gaussian noise
        noise_variance = 0.2  # Variance of the gaussian noise
        # Gaussian noise error for each sample in x
        noise = numpy.random.randn(self.x.shape[0]) * noise_variance
        # Create targets t (ground truth)
        self.t = self.nn.f(self.x) + noise

    def test_target_function(self):
        # Plot the target t versus the input x
        plt.plot(self.x, self.t, 'o', label='t')
        # Plot the initial line
        plt.plot([0, 1], [self.nn.f(0), self.nn.f(1)], 'b-', label='f(x)')
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$t$', fontsize=15)
        plt.ylim([0,2])
        plt.title('inputs (x) vs targets (t)')
        plt.grid()
        plt.legend(loc=2)
        plt.savefig("./result/target_func.png")
        plt.close()

    def test_weight_function(self):
        # Plot the cost vs the given weight w
        # Define a vector of weights for which we want to plot the cost
        self.ws = numpy.linspace(0, 4, num=100)  # weight values
        # cost_ws = numpy.vectorize(lambda w: self.cost(self.nn(self.x, w), self.t))(ws)  # cost for each weight in ws
        self.cost_ws = numpy.zeros_like(self.ws)

        for i in range (self.ws.shape[0]):
            pred = self.nn.forward(self.x, self.ws[i])
            print("pred", pred)
            self.cost_ws[i] = self.nn.cost(pred, self.t)
        # Plot
        plt.plot(self.ws, self.cost_ws, 'r-')
        plt.xlabel('$w$', fontsize=15)
        plt.ylabel('$\\xi$', fontsize=15)
        plt.title('cost vs. weight')
        plt.grid()
        plt.savefig("./result/cost_func.png")
        plt.close()

def main():
    test = test_nn()
    test.define_data()
    test.test_target_function()
    test.test_weight_function()
    test.test_gradient_descent()

if __name__ == '__main__':
    main()
