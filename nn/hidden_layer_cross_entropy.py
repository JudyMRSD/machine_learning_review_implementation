# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm # Colormaps
# Allow matplotlib to plot inside this notebook
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)


Data_Dir = "./hidden_layer_result/"
class Hidden_Layer_Cross_Entropy():
    def __init__(self):
        pass
    def rbf(self, z):
        return np.exp(-z ** 2)

    # Define the logistic function
    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    # Function to compute the hidden activations
    def hidden_activations(self, x, wh):
        return self.rbf(x * wh)

    # Define output layer feedforward
    def output_activations(self, h, wo):
        return self.logistic(h * wo - 1)

    # Define the neural network function
    def nn(self, x, wh, wo):
        return self.output_activations(self.hidden_activations(x, wh), wo)
    # Define the neural network prediction function that only returns
    #  1 or 0 depending on the predicted class
    def nn_predict(self, x, wh, wo):
        return np.around(self.nn(x, wh, wo))

class Test():
    def __init__(self):
        self.model = Hidden_Layer_Cross_Entropy()

    def define_data(self):
        # Define and generate the samples
        nb_of_samples_per_class = 10  # The number of sample in each class
        blue_mean = [0]  # The mean of the blue class
        red_left_mean = [-2]  # The mean of the red class
        red_right_mean = [2]  # The mean of the red class

        std_dev = 0.5  # standard deviation of both classes
        # Generate samples from both classes
        x_blue = np.random.randn(nb_of_samples_per_class, 1) * std_dev + blue_mean
        x_red_left = np.random.randn(nb_of_samples_per_class, 1) * std_dev + red_left_mean
        x_red_right = np.random.randn(nb_of_samples_per_class, 1) * std_dev + red_right_mean

        # Merge samples in set of input variables x, and corresponding set of
        # output variables t
        x = np.vstack((x_blue, x_red_left, x_red_right))
        t = np.vstack((np.ones((x_blue.shape[0], 1)),
                       np.zeros((x_red_left.shape[0], 1)),
                       np.zeros((x_red_right.shape[0], 1))))

        # Plot samples from both classes as lines on a 1D space
        plt.figure(figsize=(8, 0.5))
        plt.xlim(-3, 3)
        plt.ylim(-1, 1)
        # Plot samples
        plt.plot(x_blue, np.zeros_like(x_blue), 'b|', ms=30)
        plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms=30)
        plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms=30)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title('Input samples from the blue and red class')
        plt.xlabel('$x$', fontsize=15)
        plt.savefig(Data_Dir+"data.png")

    def plot_rbf(self):
        # Plot the rbf function
        z = np.linspace(-6, 6, 100)
        plt.plot(z, self.rbf(z), 'b-')
        plt.xlabel('$z$', fontsize=15)
        plt.ylabel('$e^{-z^2}$', fontsize=15)
        plt.title('RBF function')
        plt.grid()
        plt.show()

def main():
    test = Test()
    test.define_data()


if __name__ == '__main__':
    main()