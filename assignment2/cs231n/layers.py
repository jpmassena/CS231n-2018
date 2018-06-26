from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    N = x.shape[0]  # Number of example inputs N
    # Reshape N x (d_1, ..., d_k) to N x (d_1 x ... x d_k)
    X = np.reshape(x, (N, np.prod(x.shape[1:])))  # (N x D)

    out = X.dot(w) + b  # linear function W.x + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    N = x.shape[0]  # Number of example inputs N
    # Reshape N x (d_1, ..., d_k) to N x (d_1 x ... x d_k)
    X = x.reshape(N, np.prod(x.shape[1:]))  # (N x D)

    db = np.sum(dout, axis=0)  # bias gradient (M)

    dw = X.T.dot(dout)  # weight gradient (D, M)

    dx = dout.dot(w.T)  # input gradient
    dx = dx.reshape(x.shape)  # (N, d1, ..., d_k)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(0, x)  # ReLU

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    # gradient of ReLU is 1 if > 0 and 0 otherwise
    dx = dout * (np.maximum(0, x) > 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        # axis 0 to get mean and var by dimension and not by example

        # Step 1 - Batch mean: 1/N * sum(x)
        sample_mean = x.mean(0)  # (D,)

        # Step 2 - Center x around mean
        x_centered = x - sample_mean  # (N, D)

        # Step 3 & 4 - Batch var: 1/N * sum(x_centered^2)
        # Step 3 = x_centered^2 // Step 4 = 1/N * sum(step3)
        sample_var = x.var(0)  # (D,)

        # Step 5 - ~Standard Deviation = sqrt(variance)
        std = np.sqrt(sample_var + eps)

        # Step 6 & 7 - Invert the std and Normalize the batch
        # Step 6 = 1/std
        i_std = 1./std
        x_hat = x_centered * i_std

        # Step 8 & 9 - Calculate linear output
        # Step 8 = gamma * x_hat // Step 9 = step8 + beta
        out = gamma * x_hat + beta

        cache = (x_hat, gamma, x_centered, std, i_std)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        sample_normalize = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * sample_normalize + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    N, D = dout.shape

    # cache = (x_hat, gamma, x_centered, std, sample_var, eps)
    x_hat, gamma, x_centered, std, i_std = cache

    # Gradient at step 9
    dbeta = np.sum(dout, axis=0)  # (D,)

    # Gradient at step 8
    dgamma = np.sum(x_hat * dout, axis=0)  # (D,)
    d_x_hat = dout * gamma  # (N,D)

    # Gradient at step 7
    d_x_centered = d_x_hat * (i_std)  # (N,D)
    d_i_var = np.sum(d_x_hat * x_centered, axis=0)  # (D,)

    # Gradient at step 6 - We continue to go backward in the denominator
    d_std = d_i_var * (-1 / std**2)  # (D,)

    # Gradient at step 5
    d_var = d_std * (1/(2*std))  # (D,)

    # Gradient at step 4
    # var is 1/N sum of something(next step) and gradient of sum = 1*
    d_var = d_var * 1/N * np.ones((N, D))  # (N, D)

    # Gradient at step 3
    d_squared = d_var * 2 * x_centered  # (D,)

    # Gradient at step 2 - In this node, we have 2 gradients coming, so we add
    # them
    d_x1 = d_squared + d_x_centered  # (N, D)
    d_mean = -1 * np.sum(d_squared + d_x_centered, axis=0)  # (D,)

    # Gradient at step 1
    d_x2 = d_mean * 1/N * np.ones(dout.shape)

    dx = d_x1 + d_x2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x_hat, gamma, x_centered, std, i_std = cache
    N, D = dout.shape

    dbeta = dout.sum(0)  # (D,)
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)

    d_x_hat = dout * gamma  # (N,D)

    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx = (1./N) * i_std * (N * d_x_hat - np.sum(d_x_hat, axis=0) -
                           x_hat * np.sum(d_x_hat * x_hat, axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################

    x = x.T  # we only need to transpose X to keep all code from batch_norm

    # Step 1 - Batch mean: 1/N * sum(x)
    sample_mean = x.mean(0)  # (N,)

    # Step 2 - Center x around mean
    x_centered = x - sample_mean  # (D, N)

    # Step 3 & 4 - Batch var: 1/N * sum(x_centered^2)
    # Step 3 = x_centered^2 // Step 4 = 1/N * sum(step3)
    sample_var = x.var(0)  # (N,)

    # Step 5 - ~Standard Deviation = sqrt(variance)
    std = np.sqrt(sample_var + eps)  # (N,)

    # Step 6 & 7 - Invert the std and Normalize the batch
    # Step 6 = 1/std
    i_std = 1./std  # (N,)
    x_hat = x_centered * i_std  # (D,N)

    # Transpose back, now shape of xhat (N, D)
    x_hat = x_hat.T

    # Step 8 & 9 - Calculate linear output
    # Step 8 = gamma * x_hat // Step 9 = step8 + beta

    out = gamma * x_hat + beta

    cache = (x_hat, gamma, x_centered, std, i_std)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x_hat, gamma, x_centered, std, i_std = cache

    dbeta = dout.sum(0)  # (D,)
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)

    d_x_hat = dout * gamma  # (N,D)

    # Transpose dxhat and xhat back
    d_x_hat = d_x_hat.T
    x_hat = x_hat.T

    # Actually x_hat's shape is (D, N), we use notation (N, D) to let us copy
    # batch normalization backward code when computing dx without changes
    N, D = x_hat.shape

    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx = (1./N) * i_std * (N * d_x_hat - np.sum(d_x_hat, axis=0) -
                           x_hat * np.sum(d_x_hat * x_hat, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx.T, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        N, D = x.shape
        # Inverted dropout: we divide by p here instead of scaling at test time
        mask = (np.random.rand(N, D) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Unpack parameters
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H1 = int(1 + (H + 2 * pad - HH) / stride)
    W1 = int(1 + (W + 2 * pad - WW) / stride)

    out = np.zeros((N, F, H1, W1))

    # pad input with 0s in each edge equally
    pad_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):  # For each different input
        for filter_i in range(F):  # For each filter
            for r in range(H1):  # For each input row
                for c in range(W1):  # For each input column
                    # Apply convolution at input i with filter_i at row r and
                    # column c
                    out[i, filter_i, r, c] = np.sum(pad_x[
                        i, :, r*stride:r*stride+HH, c*stride:c*stride+WW] *
                        w[filter_i]) + b[filter_i]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    _, _, HH, WW = w.shape
    N, F, H, W = dout.shape

    H1 = int(1 + (H + 2 * pad - HH) / stride)
    W1 = int(1 + (W + 2 * pad - WW) / stride)

    # pad input with 0s in each edge equally
    pad_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    db = np.zeros_like(b)
    d_pad_x = np.zeros_like(pad_x)
    dw = np.zeros_like(w)

    for i in range(N):  # For each different input
        for filter_i in range(F):  # For each filter
            for r in range(H1):  # For each input row
                for c in range(W1):  # For each input column
                    # Biases gradients
                    db[filter_i] += dout[i, filter_i, r, c].sum(0)

                    # Inputs gradients
                    d_pad_x[i, :, r*stride:r*stride+HH, c*stride:c*stride+WW] \
                        += w[filter_i] * dout[i, filter_i, r, c]

                    # Weights gradients
                    dw[filter_i] += pad_x[
                        i, :, r*stride:r*stride+HH, c*stride:c*stride+WW] * \
                        dout[i, filter_i, r, c]

    # "Unpad" dx
    dx = d_pad_x[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H1 = int(1 + (H - pool_height) / stride)
    W1 = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H1, W1))

    for i in range(N):  # For each input of the "batch"
        for row in range(H1):  # For each row of the input
            for col in range(W1):  # For each column of the input
                for c in range(C):  # # For each channel of the input
                    r_s = row*stride
                    c_s = col*stride
                    # Get the max from the items inside the "crop" zone
                    out[i, c, row, col] = np.max(
                        x[i, c, r_s:r_s+pool_height, c_s:c_s+pool_width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H1 = int(1 + (H - pool_height) / stride)
    W1 = int(1 + (W - pool_width) / stride)

    dx = np.zeros_like(x)

    for i in range(N):  # For each input of the "batch"
        for row in range(H1):  # For each row of the input
            for col in range(W1):  # For each column of the input
                for c in range(C):  # # For each channel of the input
                    r_s = row*stride
                    c_s = col*stride
                    x_crop = x[i, c, r_s:r_s+pool_height, c_s:c_s+pool_width]
                    max_r, max_c = np.unravel_index(x_crop.argmax(),
                                                    x_crop.shape)
                    dx[i, c, max_r+r_s, max_c+c_s] = dout[i, c, row, col]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.shape
    # We put C as the last dimension and reshape 4D matrix to 2D
    x_t = x.transpose(0, 2, 3, 1).reshape(-1, C)
    # Calculate Batch Normalization
    x_bn, cache = batchnorm_forward(x_t, gamma, beta, bn_param)
    # Reorder the dimensions to 4D
    out = x_bn.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = dout.shape
    dout_t = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    # Calculate Batch Normalization
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_t, cache)
    # Reorder the dimensions to 4D
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape

    # Follows implementation of Fig 3: (https://arxiv.org/pdf/1803.08494.pdf)
    gn_x = x.reshape(N, G, C//G, H, W)

    # Step 1 - Batch mean: 1/N * sum(x)
    sample_mean = gn_x.mean(axis=(2, 3, 4), keepdims=True)

    # Step 2 - Center x around mean
    x_centered = gn_x - sample_mean

    # Step 3 & 4 - Batch var: 1/N * sum(x_centered^2)
    # Step 3 = x_centered^2 // Step 4 = 1/N * sum(step3)
    sample_var = gn_x.var(axis=(2, 3, 4), keepdims=True)

    # Step 5 - ~Standard Deviation = sqrt(variance)
    std = np.sqrt(sample_var + eps)

    # Step 6 & 7 - Invert the std and Normalize the batch
    # Step 6 = 1/std
    i_std = 1./std
    x_hat = x_centered * i_std

    # Reshape back
    x_hat = x_hat.reshape(N, C, H, W)

    # Step 8 & 9 - Calculate linear output
    # Step 8 = gamma * x_hat // Step 9 = step8 + beta
    out = gamma * x_hat + beta

    cache = (x_hat, gamma, i_std, G)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x_hat, gamma, i_std, G = cache
    N, C, H, W = dout.shape

    # Scale and shift are performed on the whole batch, by C
    dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)  # (1,C,1,1)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)  # (1,C,1,1)

    d_x_hat = dout * gamma  # (N, C, H, W)

    # Reshape to replicate forward pass dimensions
    g_d_x_hat = d_x_hat.reshape(N, G, C//G, H, W)
    g_x_hat = x_hat.reshape(N, G, C//G, H, W)

    # We want the dimensions we used to compute mean and var
    # This way we only need to change the dimensions indexes below from
    # the layernorm equation
    N1 = (C//G) * H * W

    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx = (1./N1) * i_std * (N1 * g_d_x_hat - np.sum(g_d_x_hat, axis=(2, 3, 4), keepdims=True) -
                            g_x_hat * np.sum(g_d_x_hat * g_x_hat, axis=(2, 3, 4), keepdims=True))
    dx = dx.reshape(N, C, H, W)
   ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
