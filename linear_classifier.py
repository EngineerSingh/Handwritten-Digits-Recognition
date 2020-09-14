import numpy as np
from svm_loss import svm_loss_vectorized
from softmax_loss import softmax_loss_vectorized

#Linear classifier class

class LinearClassifier(object):
    def __init__(self,W=None):
        self.W = W

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=128,learning_rate_decay=0.95, verbose=False):
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1  
        if self.W is None:
            # initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
            
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Run stochastic gradient descent to optimize W
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            # randomize indices
            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # perform parameter update
            self.W += - learning_rate * grad
            if verbose and it % 1000 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                # Every epoch, check train and val accuracy and decay learning rate.
            if it % 1000 == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

        return loss_history

    def predict(self, X):

        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass

class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
        
class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)