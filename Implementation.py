#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *




# In[5]:
#Standardize
def standardize(x):
    """Standardize the original data set."""
    mean=[]
    std=[]
    dims=np.shape(x)
    D=dims[1]
    for i in range (D):
        mean_x = np.mean(x[:,i])
        x[:,i] = x[:,i] - mean_x
        std_x = np.std(x[:,i])
        x[:,i]= x[:,i]/std_x
        mean.append(mean_x)
        std.append(std_x)
    return x, mean, std
#Build polynomial

def remove_outliers(x,y):
    """trying to assign outliers random non-outlier x values"""
    indices_outliers=x==-999 #return a boolean vector of size x containing 1 for each outlier
    indices_good=not indices_outliers #returns the rest 
    for i in range(x.shape[1]): #iteration over each feature
        indices_yequals1=y(y[indices_good[:,i]]==1)
        indices_ynotequals1=y(y[indices_good[:,i]]==-1)
        for index,val in indices_good[i]: #iteration over indices
             if not val: #if val of indices is false
                    if y[index]==1:
                        x[index,i]=random.choice(x[indices_yequals1,i]) #assign x to have a random x value that has the same y value
                    else:
                        x[index,i]=random.choice(x[indices_ynotequals1,i])
                       
                    
                 
                 
            

def build_poly(tx, degree):
    dims=np.shape(tx)
    n=dims[0]
    D=dims[1]
    phi=np.ones((n,D*degree+1))
    p=1;
    for i in range(1,degree*D,D):
        phi[:,i:i+D]=np.power(tx,p)
        p=p+1
    return phi

def build_poly_improved(tx, degrees,exp):
    #exp is a boolean vector of size D where 1 correspond to evaluate a exponential in the model
    #degrees is an int vector of size D where each entry corresponds to the degree of polynomial evaluated in the model 
    dims=np.shape(tx)
    n=dims[0]
    D=dims[1]
    No_Cols=1+np.sum((degrees))+np.sum((exp)) #No. columns in phi
    phi=np.ones((n,int(No_Cols))) #preallocate phi
    p=1 #initialize power at 1
    i=1 #initialize position at 1 because column 0 is a constant term 
    while (i<No_Cols):
        tx_polyadd=tx[:,degrees>=p]
        tx_expadd=tx[:,exp==i] #will only be evaluated once
        elements=np.shape(tx_polyadd)[1]+np.shape(tx_expadd)[1]
        phi[:,i:i+elements]=np.concatenate((np.power(tx_polyadd,p),np.exp(tx_expadd)),axis=1)
        p=p+1
        i=i+elements
    return phi


#Compute Loss
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean((e)**2)


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    y_pred = predict_labels(w,tx) #returns -1 and +1
    
    e =(y - y_pred)
    return calculate_mse(e)


# In[6]:


# Calculate gradient
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    y_pred=predict_labels(w,tx)
    e = (y - y_pred);
    grad = -tx.T.dot(e) / len(e) #x transposed times error (with sign)
    return grad, e


# In[7]:
#Split Data into test and training
import numpy as np


def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(ratio * num_row)
    index_tr = indices[:index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr,:]
    x_te = x[index_te,:]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


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
            print("SGD({bi}/{ti}): loss={l} ".format(bi=n_iter, ti=max_iters - 1, l=loss))
            for i in range (len(w)):
                print("w({index}): {weight} ". format(index=i, weight=w[i]))
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
    lambda_prime=lambda_*2*np.shape(tx)[0]
    D=tx.shape[1]
    a=np.transpose(tx).dot(tx)+lambda_prime*np.identity(D)
    b=tx.T.dot(y)
    w_opt= np.linalg.solve(a,b)
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
def print_ws(w):
    """Print ws"""
    for i in range (len(w)):
          print("w({index}): {weight} ". format(index=i, weight=w[i]))
    print ("\n")

# In [13]
#Indices: Create an array with k_fold rows containing n/k_fold indices
def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
# In [14]
#K-fold Cross Validation
def cross_validation(y, phi, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    phi_te=phi[k_indices[k,:],:]
    y_te=y[k_indices[k,:]]
    rest_indeces=np.delete(k_indices,k,0)
    phi_tr=phi[rest_indeces.flatten(),:]
    y_tr=y[rest_indeces.flatten()]
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    loss_tr,w_tr=ridge_regression(y_tr,phi_tr,lambda_)
    
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_te=compute_loss(y_te,phi_te,w_tr)
    # raise NotImplementedError
    return loss_tr, loss_te,w_tr
      
#In [15]
#This program will iterate over different lambdas and find the one that gives the minimum test loss    
def min_loss_crossvalidation(tx,y):
        """We want to find the minimum lambda by changing it and iterating  """
   