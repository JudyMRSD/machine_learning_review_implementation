# reference:
# https://github.com/peterroelants/peterroelants.github.io/blob/master/notebooks/RNN_implementation/rnn_implementation_part01.ipynb

# implementation of forward and backward of rnn with linear activation function
import numpy as np

class rnn():
    def __init__(self, X):
        self.X = X


    # Define the forward step functions
    def update_state(self, xk, sk, wx, wRec):
        """
        Compute state k from the previous state (sk) and current input (xk),
        by use of the input weights (wx) and recursive weights (wRec).
        """
        return xk * wx + sk * wRec

    def forward_states(self, wx, wRec):
        """
        Unfold the network and compute all state activations given the input X,
        and input weights (wx) and recursive weights (wRec).
        Return the state activations in a matrix, the last column S[:,-1] contains the
        final activations.
        """
        # Initialise the matrix that holds all states for all input sequences.
        # The initial state S[:, 0], 0th column in S, is set to zeros.
        S = np.zeros((self.X.shape[0], self.X.shape[1]+1))
        # Use the recurrence relation defined by update_state to update the
        #  states trough time.
        for k in range(0, self.X.shape[1]):
            # S[k] = S[k-1] * wRec + X[k] * wx
            S[:,k+1] = self.update_state(self.X[:,k], S[:,k], wx, wRec)
        return S


    def cost(self, y, t):
        """
        Return the MSE between the targets t and the outputs y.
        """
        nb_of_samples = self.X.shape[0]
        return ((t - y)**2).sum() / nb_of_samples

    def output_gradient(self, y, t):
        """
        Compute the gradient of the MSE cost function with respect to the output y.
        """
        return 2.0 * (y - t) / self.X.shape[0]

    def backward_gradient(self, S, grad_out, wRec):
        """
        Backpropagate the gradient computed at the output (grad_out) through the network.
        Accumulate the parameter gradients for wX and wRec by for each layer by addition.
        Return the parameter gradients as a tuple, and the gradients at the output of each layer.
        """
        # Initialise the array that stores the gradients of the cost with respect to the states.
        grad_over_time = np.zeros((self.X.shape[0], self.X.shape[1] + 1))
        grad_over_time[:, -1] = grad_out
        # Set the gradient accumulations to 0
        wx_grad = 0
        wRec_grad = 0
        # loop through number of samples
        for k in range(self.X.shape[1], 0, -1):
            # Compute the parameter gradients and accumulate the results.
            wx_grad += np.sum(grad_over_time[:, k] * self.X[:, k - 1])
            wRec_grad += np.sum(grad_over_time[:, k] * S[:, k - 1])
            # Compute the gradient at the output of the previous layer
            grad_over_time[:, k - 1] = grad_over_time[:, k] * wRec
        return (wx_grad, wRec_grad), grad_over_time

class test_tool():
    def __init__(self):
        self.create_dataset()
        self.rnn_object = rnn(self.X)
        self.params = [1.2, 1.2]  # [wx, wRec]

    def create_dataset(self):
        # Create dataset
        self.nb_of_samples = 20
        sequence_len = 10
        # Create the sequences
        # The input data X used in this example consists of 20 binary sequences of 10 timesteps each.
        # Each input sequence is generated from a uniform random distribution which is rounded to 0 or 1.
        self.X = np.zeros((self.nb_of_samples, sequence_len))
        for row_idx in range(self.nb_of_samples):
            self.X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
        # Create the targets for each sequence
        # many to many network
        # The output targets t are the number of times '1' occurs in the sequence,
        # which is equal to the sum of that sequence since the sequence is binary.
        self.t = np.sum(self.X, axis=1)
        # print(self.X)
        # print(t)

    def test_forward(self):
        # Perform gradient checking
        # Set the weight parameters used during gradient checking

        # Set the small change to compute the numerical gradient
        eps = 1e-7
        # Compute the backprop gradients
        self.S = self.rnn_object.forward_states(self.params[0], self.params[1])
        # print("s", S)

    def test_backward(self):
        grad_out = self.rnn_object.output_gradient(self.S[:, -1], self.t)
        backprop_grads, grad_over_time = self.rnn_object.backward_gradient(self.S, grad_out, self.params[1])
        print("grad_out", grad_out)
        print("backprop_grads, grad_over_time ", backprop_grads, grad_over_time )

def main():
    test_obj = test_tool()
    test_obj.create_dataset()
    test_obj.test_forward()
    test_obj.test_backward()

if __name__ == "__main__":
    main()