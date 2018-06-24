from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(
            scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(
            scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        # first layer is affine + relu
        out1, cache1 = affine_relu_forward(
            X, self.params['W1'], self.params['b1'])

        # second layer is affine
        scores, cache2 = affine_forward(
            out1, self.params['W2'], self.params['b2'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, d_out = softmax_loss(scores, y)  # compute loss and its gradient

        # L2 regularization
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))

        # do backprop
        # 2nd layer
        d_x2, d_w2, d_b2 = affine_backward(d_out, cache2)

        # 1st layer
        d_x1, d_w1, d_b1 = affine_relu_backward(d_x2, cache1)

        d_w2 += self.reg*W2
        d_w1 += self.reg*W1

        grads = {'W1': d_w1, 'b1': d_b1, 'W2': d_w2, 'b2': d_b2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a BatchNorm
    and then a relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Shift and scale for batch normalization layer
    - bn_param: Dictionary with batch normalization parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_layernorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a LayerNorm
    and then a relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Shift and scale for layer normalization layer
    - bn_param: Dictionary with layer normalization parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = layernorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    # print("relu_backward")
    da = relu_backward(dout, relu_cache)
    # print("batchnorm_backward")
    d_bn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    # print("affine_backward")
    dx, dw, db = affine_backward(d_bn, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    d_bn, dgamma, dbeta = layernorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(d_bn, fc_cache)
    return dx, dw, db, dgamma, dbeta


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        # this 'joins' all the dims in the order that we want
        layer_dims = np.hstack([input_dim, hidden_dims, num_classes])

        # Setup the layers inputs/outputs
        for layer in range(self.num_layers):
            self.params['W'+str(layer+1)] = np.random.normal(
                scale=weight_scale,
                size=(layer_dims[layer], layer_dims[layer+1]))
            self.params['b'+str(layer+1)] = np.zeros(layer_dims[layer+1])
            if self.normalization is not None and (
                    layer < self.num_layers - 1):
                self.params['gamma'+str(layer+1)] = np.ones(
                    layer_dims[layer+1])
                self.params['beta'+str(layer+1)] = np.zeros(
                    layer_dims[layer+1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        x = X  # layer input
        grad_caches = []  # intermediate results to compute gradient later
        drop_caches = []
        # For each layer, except the last, do affine-relu forward pass. Store
        # the intermediate results in the Cache
        for layer in range(self.num_layers-1):
            W = self.params['W'+str(layer+1)]
            b = self.params['b'+str(layer+1)]
            if self.normalization is None:
                # print("af_relu_forward {} with W{}".format(layer,layer+1))
                x, cache = affine_relu_forward(x, W, b)
            else:
                gamma = self.params['gamma'+str(layer+1)]
                beta = self.params['beta'+str(layer+1)]

                if self.normalization == 'batchnorm':
                    # print("af_bn_relu_forward {} with W{} and gamma{}".format(layer,layer+1,layer+1))
                    x, cache = affine_batchnorm_relu_forward(
                        x, W, b, gamma, beta, self.bn_params[layer])
                else:
                    # print("af_ln_relu_forward {} with W{}".format(layer,layer+1))
                    x, cache = affine_layernorm_relu_forward(
                        x, W, b, gamma, beta, self.bn_params[layer])
            if self.use_dropout:
                x, drop_cache = dropout_forward(x, self.dropout_param)
                drop_caches.append(drop_cache)

            grad_caches.append(cache)

        # Calculate the output scores with the last layer's weights
        W = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        # print("affine_forward {} with W{}".format(self.num_layers-1,self.num_layers))
        scores, cache = affine_forward(x, W, b)
        grad_caches.append(cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, d_out = softmax_loss(scores, y)  # compute loss and its gradient

        # L2 regularization WRT last layer weights
        loss += 0.5 * self.reg * np.sum(
            self.params['W'+str(self.num_layers)]**2)

        # Gradients of the last affine layer
        # print("affine_backward cache{}".format(len(grad_caches)))
        last_d_x, last_d_w, last_d_b = affine_backward(d_out, grad_caches[-1])
        # print("update W{}".format(self.num_layers))
        grads['W'+str(self.num_layers)] = last_d_w + self.reg * \
            self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = last_d_b

        d_out = last_d_x

        # For each layer starting in the before-last and going backwards
        for layer in range(self.num_layers-2, -1, -1):
            # Do L2 regularization
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(layer+1)]**2)
            # print("cache{}".format(layer+1))
            if self.normalization is None:
                if self.use_dropout:
                    d_out = dropout_backward(d_out, drop_caches[layer])
                d_x, d_w, d_b = affine_relu_backward(d_out, grad_caches[layer])
            else:
                if self.normalization == 'batchnorm':
                    if self.use_dropout:
                        d_out = dropout_backward(d_out, drop_caches[layer])
                    d_x, d_w, d_b, d_gamma, d_beta = \
                        affine_batchnorm_relu_backward(
                            d_out, grad_caches[layer])
                else:
                    if self.use_dropout:
                        d_out = dropout_backward(d_out, drop_caches[layer])
                    d_x, d_w, d_b, d_gamma, d_beta = \
                        affine_layernorm_relu_backward(
                            d_out, grad_caches[layer])
                # print("update gamma{}".format(layer+1))
                grads['gamma'+str(layer+1)] = d_gamma
                grads['beta'+str(layer+1)] = d_beta
            # print("update W{}".format(layer+1))
            grads['W'+str(layer+1)] = d_w + self.reg * \
                self.params['W'+str(layer+1)]
            grads['b'+str(layer+1)] = d_b

            d_out = d_x

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
