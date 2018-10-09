#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


# In[5]:


#Compute Loss
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    y_pred = predict_labels(w,tx) #returns -1 and +1
    
    e = y - y_pred
    return calculate_mse(e)


# In[6]:


# Calculate gradient
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    y_pred=predict_labels(w,tx)
    e = y - y_pred;
    grad = -tx.T.dot(e) / len(e) #x transposed times error (with sign)
    return grad, e


# In[7]:


# Least Squares GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Return weights and loss using the gradient descent
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


# In[8]:


#Stochastic Gradient Descent
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def least_squares_SGD(y, tx, initial_w,batch_size,max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("SGD({bi}/{ti}): loss={l} ".format(
              bi=n_iter, ti=max_iters - 1, l=loss)
            for i in range(len(w))
                print("w({index}): {weight} ". format(index=i, weight=w[i])
            print ("\n")
    return losses, ws


# In[9]:


#Normal Equations: least Squares
def least_squares(y, tx):
    from numpy.linalg import inv
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    gram=np.transpose(tx).dot(tx)
    w_opt=inv(gram).dot(np.transpose(tx).dot(y))
    loss=compute_loss(y,tx,w_opt)
    return loss,w_opt
    #raise NotImplementedError


# In[10]:


#Ridge Regression:
def ridge_regression(y, tx, lambda_):
    #"""implement ridge regression."""
    from numpy.linalg import inv
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    lambda_prime=lambda_*2*len(tx)
    D=min(tx.shape)
    gram=np.transpose(tx).dot(tx)+lambda_prime*np.identity(D)
    w_opt= inv(gram).dot(np.transpose(tx).dot(y))
    loss=compute_loss(y,tx,w_opt)
    return loss,w_opt


# In[11]:


#Plot Data with respect to specific feature
def plot_data(y,tx,feature):
    """plot the fitted curve."""
    xval=tx[:,feature-1] #Let first feature be numbered 1
    plt.scatter(xval, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(xval) - 0.1, max(xval) + 0.1, 0.1)
    #tx = build_poly(xvals, degree)
    #f = tx.dot(weights)
    #ax.plot(xvals, f)
    plt.set_xlabel("x "+str(feature))
    plt.set_ylabel("y")
    plt.set_title("y vs. feature # " + str(feature))


# In[ ]:




