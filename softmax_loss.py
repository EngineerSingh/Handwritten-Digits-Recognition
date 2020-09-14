import numpy as np

def softmax_loss_vectorized(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # loss
    # score: N by C matrix containing class scores
    scores = X.dot(W)
    scores -= scores.max()
    scores = np.exp(scores)
    scores_sums = np.sum(scores, axis=1)
    cors = scores[range(num_train), y]
    loss = cors / scores_sums
    loss = -np.sum(np.log(loss))/num_train + reg * np.sum(W * W)
    # grad
    s = np.divide(scores, scores_sums.reshape(num_train, 1))
    s[range(num_train), y] = - (scores_sums - cors) / scores_sums
    dW = X.T.dot(s)
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW