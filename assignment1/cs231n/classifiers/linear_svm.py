import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        diff_count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                diff_count += 1
                # gradient for incorrect classes
                dW[:, j] += X[i]
                loss += margin
        # gradient for the correct class
        dW[:, y[i]] += -diff_count*X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)  # regularization penalty is the L2 norm
    dW += reg * 2 * W

    ###########################################################################
    # TODO:                                                                   #
    # Compute the gradient of the loss function and store it dW.              #
    # Rather that first computing the loss and then computing the derivative, #
    # it may be simpler to compute the derivative at the same time that the   #
    # loss is being computed. As a result you may need to modify some of the  #
    # code above to compute the gradient.                                     #
    ###########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    delta = 1.0
    ###########################################################################
    # TODO:                                                                   #
    # Implement a vectorized version of the structured SVM loss, storing the  #
    # result in loss.                                                         #
    ###########################################################################
    # scores = W.dot(x)
    # # compute the margins for all classes in one vector operation
    # margins = np.maximum(0, scores - scores[y] + delta)
    # # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # # to ignore the y-th position and only consider margin on max wrong class
    # margins[y] = 0
    # loss_i = np.sum(margins)

    scores = np.zeros_like(X)
    scores = X.dot(W)  # (N,C)
    correct_c_score = scores[np.arange(num_train), y].reshape(-1, 1)  # (N,1)
    
    margins = np.maximum(0, scores - correct_c_score + delta)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    ###########################################################################
    # TODO:                                                                   #
    # Implement a vectorized version of the gradient for the structured SVM   #
    # loss, storing the result in dW.                                         #
    #                                                                         #
    # Hint: Instead of computing the gradient from scratch, it may be easier  #
    # to reuse some of the intermediate values that you used to compute the   #
    # loss.                                                                   #
    ###########################################################################

    # column maps to class, row maps to sample; a value v in X_mask[i, j]
    # adds a row sample i to column class j with multiple of v
    margins[margins > 0] = 1
    # for each sample, find the total number of classes where margin > 0
    incorrect_counts = np.sum(margins, axis=1)

    margins[np.arange(num_train), y] -= incorrect_counts
    dW = X.T.dot(margins)

    dW /= num_train
    dW += reg * 2 * W
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, dW
