# Python imports# Pytho
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from matplotlib import cm # Colormaps
# Allow matplotlib to plot inside this notebook
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)

class logistic_regression():
    def __init__(self):
        pass

    def logistic(self, z):
        return 1/(1+np.exp(-z))


    # Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
    def forward(self, x, w):
        return self.logistic(x.dot(w.T))
    # Define the neural network prediction function that only returns
    #  1 or 0 depending on the predicted class
    def nn_predict(self, x, w):
        pred = self.forward(x, w)
        # return 1 when pred>0.5
        return np.around(pred)

    # cross entopy loss function
    # E(t_i, y_i) = -t_i *log(y_i) - (1-t_i)*log(1-y_i)
    def cost(self,y,t):
        cost1 = np.multiply(t, np.log(y))
        cost2 = np.multiply((1 - t), np.log(1 - y))
        return - np.sum(cost1 + cost2)

    # gradient
    # dE/dw
    def gradient(self, w, x, t):
        return (self.forward(x, w) - t).T * x

    # define the update function delta w which returns the
    #  delta w for each weight in a vector
    def delta_w(self, w_k, x, t, learning_rate):
        return learning_rate * self.gradient(w_k, x, t)


    # gradient descent update
    def gradient_descent_update(self, w, X, t, learning_rate):

        # Start the gradient descent updates and plot the iterations
        nb_of_iterations = 10  # Number of gradient descent updates
        w_iter = [w]  # List to store the weight values over the iterations
        for i in range(nb_of_iterations):
            dw = self.delta_w(w, X, t, learning_rate)  # Get the delta w update
            w = w - dw  # Update the weights
            w_iter.append(w)  # Store the weights for plotting
        return w_iter

class test_logistic_regression():
    def __init__(self):
        self.logistic_reg = logistic_regression()

    # two classes:  blue (t=1) and red (t=0)
    # self.X is a N x 2 matrix of individual input samples x_i
    # self.t is a N x 1 matrix of corresponding groundtruth values
    def define_data(self):
        # Define and generate the samples
        nb_of_samples_per_class = 20  # The number of sample in each class
        red_mean = [-1, 0]  # The mean of the red class
        blue_mean = [1, 0]  # The mean of the blue class
        std_dev = 1.2  # standard deviation of both classes
        # Generate samples from both classes
        self.x_red = np.random.randn(nb_of_samples_per_class, 2) * std_dev + red_mean
        self.x_blue = np.random.randn(nb_of_samples_per_class, 2) * std_dev + blue_mean

        # Merge samples in set of input variables x, and corresponding set of output variables t
        self.X = np.vstack((self.x_red, self.x_blue))
        self.t = np.vstack((np.zeros((nb_of_samples_per_class, 1)), np.ones((nb_of_samples_per_class, 1))))

        # Plot both classes on the x1, x2 plane
        plt.plot(self.x_red[:, 0], self.x_red[:, 1], 'ro', label='class red')
        plt.plot(self.x_blue[:, 0], self.x_blue[:, 1], 'bo', label='class blue')
        plt.grid()
        plt.legend(loc=2)
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.axis([-4, 4, -4, 4])
        plt.title('red vs. blue classes in the input space')
        plt.savefig("./result/logistic_regression_data.png")
        plt.close()
    def vis_cost_surface(self):
        # Plot the cost in function of the weights
        # Define a vector of weights for which we want to plot the cost
        nb_of_ws = 100  # compute the cost nb_of_ws times in each dimension
        ws1 = np.linspace(-5, 5, num=nb_of_ws)  # weight 1
        ws2 = np.linspace(-5, 5, num=nb_of_ws)  # weight 2
        self.ws_x, self.ws_y = np.meshgrid(ws1, ws2)  # generate grid
        self.cost_ws = np.zeros((nb_of_ws, nb_of_ws))  # initialize cost matrix
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_ws):
            for j in range(nb_of_ws):

                self.cost_ws[i, j] = self.logistic_reg.cost(self.logistic_reg.forward(self.X,np.asmatrix([self.ws_x[i, j],self.ws_y[i, j]])),
                                                            self.t)
        # Plot the cost function surface
        plt.contourf(self.ws_x, self.ws_y, self.cost_ws, 20, cmap=cm.pink)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('$\\xi$', fontsize=15)
        plt.xlabel('$w_1$', fontsize=15)
        plt.ylabel('$w_2$', fontsize=15)
        plt.title('Cost function surface')
        plt.grid()
        plt.savefig("./result/logistic_regression_cost_function_surface.png")
        plt.close()
    def test_gradient_descent(self):

        # Set the initial weight parameter
        self.w = np.asmatrix([-4, -2])
        # Set the learning rate
        learning_rate = 0.05
        w_iter = self.logistic_reg.gradient_descent_update(self.w, self.X, self.t, learning_rate)

        # Plot the first weight updates on the error surface
        # Plot the error surface
        plt.contourf(self.ws_x, self.ws_y, self.cost_ws, 20, alpha=0.9, cmap=cm.pink)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('cost')

        # Plot the updates
        for i in range(1, 4):
            w1 = w_iter[i - 1]
            w2 = w_iter[i]
            # Plot the weight-cost value and the line that represents the update
            plt.plot(w1[0, 0], w1[0, 1], 'bo')  # Plot the weight cost value
            plt.plot([w1[0, 0], w2[0, 0]], [w1[0, 1], w2[0, 1]], 'b-')
            plt.text(w1[0, 0] - 0.2, w1[0, 1] + 0.4, '$w({})$'.format(i), color='b')
        w1 = w_iter[3]
        # Plot the last weight
        plt.plot(w1[0, 0], w1[0, 1], 'bo')
        plt.text(w1[0, 0] - 0.2, w1[0, 1] + 0.4, '$w({})$'.format(4), color='b')
        # Show figure
        plt.xlabel('$w_1$', fontsize=15)
        plt.ylabel('$w_2$', fontsize=15)
        plt.title('Gradient descent updates on cost surface')
        plt.grid()
        plt.savefig("./result/logistic_regression_gradient_descent.png")
        plt.close()

    def visualize_classifier(self):
        nb_of_xs = 200
        xs1 = np.linspace(-4, 4, num=nb_of_xs)
        xs2 = np.linspace(-4, 4, num=nb_of_xs)
        xx, yy = np.meshgrid(xs1, xs2)  # create the grid
        # Initialize and fill the classification plane
        classification_plane = np.zeros((nb_of_xs, nb_of_xs))
        for i in range(nb_of_xs):
            for j in range(nb_of_xs):
                classification_plane[i, j] = self.logistic_reg.nn_predict(np.asmatrix([xx[i, j], yy[i, j]]), self.w)
        # Create a color map to show the classification colors of each grid point
        cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])

        # Plot the classification plane with decision boundary and input samples
        plt.contourf(xx, yy, classification_plane, cmap=cmap)
        plt.plot(self.x_red[:, 0], self.x_red[:, 1], 'ro', label='target red')
        plt.plot(self.x_blue[:, 0], self.x_blue[:, 1], 'bo', label='target blue')
        plt.grid()
        plt.legend(loc=2)
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.title('red vs. blue classification boundary')
        plt.savefig("./result/visualize_logistic_regression_boundary.png")

def main():
    print("hi")
    test = test_logistic_regression()
    test.define_data()
    test.vis_cost_surface()
    test.test_gradient_descent()
    test.visualize_classifier()


if __name__ == '__main__':
    main()

