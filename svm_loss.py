import numpy as np

def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    # s: A numpy array of shape (N, C) containing scores
    s = X.dot(W)
    # read correct scores into a column array of height N
    correct_score = s[list(range(num_train)), y]
    correct_score = correct_score.reshape(num_train, -1)
    # subtract correct scores from score matrix and add margin
    s += 1 - correct_score
    # make sure correct scores themselves don't contribute to loss function
    s[list(range(num_train)), y] = 0
    # construct loss function
    loss = np.sum(np.fmax(s, 0)) / num_train
    loss += reg * np.sum(W * W)

    X_mask = np.zeros(s.shape)
    X_mask[s > 0] = 1
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW