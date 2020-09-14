import numpy as np

#Neural Network Class

class TwoLayerNet(object):
    
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    def set_params(self,W1,b1,W2,b2):
        try:
            del self.params
        except:
            pass
        self.params = {}
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
    def loss(self, X, y=None, reg=0.0):
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        # Using ReLUs as the Activation Function
        H1 = np.maximum(0, np.dot(X, W1) + b1)
        H2 = np.dot(H1, W2) + b2
        scores = H2

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        num_train = X.shape[0]
        scores -= scores.max()
        scores = np.exp(scores)
        scores_sums = np.sum(scores, axis=1)
        cors = scores[range(num_train), y]
        loss = cors / scores_sums
        loss = -np.sum(np.log(loss)) / num_train\
                + reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # Backward pass: compute gradients
        grads = {}
        s = np.divide(scores, scores_sums.reshape(num_train, 1))
        s[range(num_train), y] = - (scores_sums - cors) / scores_sums
        s /= num_train
        dW2 = H1.T.dot(s)
        # db2 = np.ones((1, num_train)).dot(s)
        db2 = np.sum(s, axis=0)
        hidden = s.dot(W2.T)
        hidden[H1 == 0] = 0
        dW1 = X.T.dot(hidden)
        # db1 = np.ones((1, num_train)).dot(hidden)
        db1 = np.sum(hidden, axis=0)
        grads['W2'] = dW2 + 2 * reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + 2 * reg * W1
        grads['b1'] = db1
        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # randomize indices
            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']
            if verbose and it % 1000 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
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

    def predict(self, X):
        y_pred = None
        y_pred = np.argmax(np.dot(np.maximum(0, X.dot(self.params['W1'])\
                + self.params['b1']), self.params['W2']) \
                + self.params['b2'], axis=1)
        return y_pred