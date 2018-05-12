# -*- coding: utf-8 -*-
"""
Created on Thu May  3 01:15:17 2018

@author: skhawy
"""

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import time

ratings = np.load("ratings.npy")

# We have triplets of (user, restaurant, rating).
#print(ratings)

M=sp.csr_matrix((ratings[:,2], (ratings[:,0], ratings[:,1])), shape=(337867, 5899), dtype=np.int64)

def cold_start_preprocessing(matrix, min_entries):
    """
    Recursively removes rows and columns from the input matrix which have less than min_entries nonzero entries.
    
    Parameters
    ----------
    matrix      : sp.spmatrix, shape [N, D]
                  The input matrix to be preprocessed.
    min_entries : int
                  Minimum number of nonzero elements per row and column.

    Returns
    -------
    matrix      : sp.spmatrix, shape [N', D']
                  The pre-processed matrix, where N' <= N and D' <= D
        
    """
    #print("Shape before: {}".format(matrix.shape))
    
    ### YOUR CODE HERE ###
    def drcsr(mat, indices):
        indices = list(indices)
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[indices] = False
        return mat[mask]

    def dccsr(mat, indices):
        indices = list(indices)
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[indices] = False
        return mat[:,mask]
    
    nnz = matrix>0
    col_ind=np.where(nnz.sum(0).A1 <= min_entries)[0]
    matrix=dccsr(matrix, col_ind)
    nnz = matrix>0
    row_ind=np.where(nnz.sum(1).A1 <= min_entries)[0]
    matrix=drcsr(matrix,row_ind)
    nnz = matrix>0
    if (nnz.sum(0).A1 > min_entries).all() != True:
        return cold_start_preprocessing(matrix, 10)
    if (nnz.sum(1).A1 > min_entries).all() != True:
        return cold_start_preprocessing(matrix, 10)
        
    
    #print("Shape after: {}".format(matrix.shape))
    nnz = matrix>0
    assert (nnz.sum(0).A1 > min_entries).all() # sums along the rows, and checks if all entries is > min_entries
    assert (nnz.sum(1).A1 > min_entries).all()
    return matrix

M = cold_start_preprocessing(M, 10)
def shift_user_mean(matrix):
    """
    Subtract the mean rating per user from the non-zero elements in the input matrix.
    
    Parameters
    ----------
    matrix : sp.spmatrix, shape [N, D]
             Input sparse matrix.
    Returns
    -------
    matrix : sp.spmatrix, shape [N, D]
             The modified input matrix.
    
    user_means : np.array, shape [N, 1]
                 The mean rating per user that can be used to recover the absolute ratings from the mean-shifted ones.

    """
    
    ### YOUR CODE HERE ###
    sum_cols=matrix.sum(1).A1 
    nnz_counts_each_row=np.diff(matrix.indptr) # indptr follwos the relation indptr[i]=indptr[i-1]+ #nnz elements in row (i-1)

    #user_means0 = sum_cols/nnz_counts_each_row
    #print("user_means0", user_means0)
    nnz_counts_each_row[nnz_counts_each_row ==0]=1
    user_means = sum_cols/nnz_counts_each_row
    
    #print("user_means", user_means)

    #user_means = np.divide(sum_cols, nnz_counts_each_row, out=np.zeros_like(sum_cols), where=nnz_counts_each_row!=0)
    #user_means=user_means.reshape(11275,1)
    tmp= matrix.copy()
    tmp.data= np.ones_like(tmp.data)
    diag_mu=sp.diags(user_means,0)
    matrix = matrix - (diag_mu*tmp)
    assert np.all(np.isclose(matrix.mean(1), 0))
    return matrix, user_means

def split_data(matrix, n_validation, n_test):
    """
    Extract validation and test entries from the input matrix. 
    
    Parameters
    ----------
    matrix          : sp.spmatrix, shape [N, D]
                      The input data matrix.
    n_validation    : int
                      The number of validation entries to extract.
    n_test          : int
                      The number of test entries to extract.

    Returns
    -------
    matrix_split    : sp.spmatrix, shape [N, D]
                      A copy of the input matrix in which the validation and test entries have been set to zero.
    
    val_idx         : tuple, shape [2, n_validation]
                      The indices of the validation entries.
    
    test_idx        : tuple, shape [2, n_test]
                      The indices of the test entries.
    
    val_values      : np.array, shape [n_validation, ]
                      The values of the input matrix at the validation indices.
                      
    test_values     : np.array, shape [n_test, ]
                      The values of the input matrix at the test indices.

    """
    '''
    ### YOUR CODE HERE ###
    matrix_split = matrix.copy()
    val_idx_row  = matrix_split.nonzero()[0][-n_validation:]
    val_idx_col  = matrix_split.nonzero()[1][-n_validation:]
    val_idx      = list(zip(val_idx_row, val_idx_col))
    val_values   = matrix_split.data[-n_validation:].copy()
    #setting elements to zero 
    #print("before",val_values)
    
    matrix_split.data[-n_validation:]=np.repeat(0,n_validation)
    #print("after",val_values)

    #test set
    test_idx_row  = matrix_split.nonzero()[0][:n_test]
    test_idx_col  = matrix_split.nonzero()[1][:n_test]
    test_idx      = list(zip(test_idx_row, test_idx_col))
    test_values   = matrix_split.data[:n_test].copy()
    #print("before",test_values)

    #setting elements to zero 
    matrix_split.data[:n_test]=np.repeat(0,n_test)
    #print("after",test_values)
    '''
    matrix_split  = matrix.copy()
    rand_idx      = np.random.permutation(M.nnz)
    
    rand_val_rows = M.nonzero()[0][rand_idx[:200]]
    rand_val_cols = M.nonzero()[1][rand_idx[:200]]
    val_idx       = list(zip(rand_val_rows,rand_val_cols))
    val_values    = matrix[rand_val_rows,rand_val_cols].copy().A1
    matrix_split[rand_val_rows,rand_val_cols]=0
    
    rand_test_rows = M.nonzero()[0][rand_idx[200:400]]
    rand_test_cols = M.nonzero()[1][rand_idx[200:400]]
    test_idx       = list(zip(rand_test_rows,rand_test_cols))
    test_values    = matrix[rand_test_rows,rand_test_cols].copy().A1
    matrix_split[rand_test_rows,rand_test_cols]=0
    
    matrix_split.eliminate_zeros()
    return matrix_split, val_idx, test_idx, val_values, test_values
#M_shifted, user_means=shift_user_mean(M)
    
n_validation = 200
n_test = 200
#M_shifted, user_means = shift_user_mean(M)

# Split data
M_train, val_idx, test_idx, val_values, test_values = split_data(M, n_validation, n_test)

nonzero_indices =  np.asarray(M.nonzero()).T # shape [nnz,2]
# Remove user means.
M_shifted, user_means = shift_user_mean(M_train)

# Apply the same shift to the validation and test data.
tmp_val_values_shifted  = val_values  - np.mean(val_values)      ### YOUR CODE HERE ###
val_values_shifted=[]
it_index=0
for row,col in val_idx:
    orig_mean=user_means[row]
    val_values_shifted.append(val_values[it_index]-orig_mean)
    it_index+=1
test_values_shifted = test_values - np.mean(test_values) ### YOUR CODE HERE ##

def initialize_Q_P(matrix, k, init='random'):
    """
    Initialize the matrices Q and P for a latent factor model.
    
    Parameters
    ----------
    matrix : sp.spmatrix, shape [N, D]
             The matrix to be factorized.
    k      : int
             The number of latent dimensions.
    init   : str in ['svd', 'random'], default: 'random'
             The initialization strategy. 'svd' means that we use SVD to initialize P and Q, 'random' means we initialize
             the entries in P and Q randomly in the interval [0, 1).

    Returns
    -------
    Q : np.array, shape [N, k]
        The initialized matrix Q of a latent factor model.

    P : np.array, shape [k, D]
        The initialized matrix P of a latent factor model.
    """

    #np.random.seed(158)
    if init == 'svd':
    ### YOUR CODE HERE ###
        u, s, VT =svds(matrix.asfptype(),k)
        Q=u.dot(s)
        P=VT
    elif init == 'random':
        Q= np.random.normal(0.5,0.5, size=(matrix.shape[0], k))
        P= np.random.normal(0.5,0.5, size=(k, matrix.shape[1]))
    ### YOUR CODE HERE ###
    else:
        raise ValueError
        
    assert Q.shape == (matrix.shape[0], k)
    assert P.shape == (k, matrix.shape[1])
    return Q, P

def latent_factor_alternating_optimization(M, non_zero_idx, k, val_idx, val_values,
                                           reg_lambda, max_steps=100, init='random',
                                           log_every=1, patience=10, eval_every=1):
    """
    Perform matrix factorization using alternating optimization. Training is done via patience,
    i.e. we stop training after we observe no improvement on the validation loss for a certain
    amount of training steps. We then return the best values for Q and P oberved during training.
    
    Parameters
    ----------
    M                 : sp.spmatrix, shape [N, D]
                        The input matrix to be factorized.
                      
    non_zero_idx      : np.array, shape [nnz, 2]
                        The indices of the non-zero entries of the un-shifted matrix to be factorized. 
                        nnz refers to the number of non-zero entries. Note that this may be different
                        from the number of non-zero entries in the input matrix M, e.g. in the case
                        that all ratings by a user have the same value.
    
    k                 : int
                        The latent factor dimension.
    
    val_idx           : tuple, shape [2, n_validation]
                        Tuple of the validation set indices.
                        n_validation refers to the size of the validation set.
                      
    val_values        : np.array, shape [n_validation, ]
                        The values in the validation set.
                      
    reg_lambda        : float
                        The regularization strength.
                      
    max_steps         : int, optional, default: 100
                        Maximum number of training steps. Note that we will stop early if we observe
                        no improvement on the validation error for a specified number of steps
                        (see "patience" for details).
                      
    init              : str in ['random', 'svd'], default 'random'
                        The initialization strategy for P and Q. See function initialize_Q_P for details.
    
    log_every         : int, optional, default: 1
                        Log the training status every X iterations.
                    
    patience          : int, optional, default: 10
                        Stop training after we observe no improvement of the validation loss for X evaluation
                        iterations (see eval_every for details). After we stop training, we restore the best 
                        observed values for Q and P (based on the validation loss) and return them.
                      
    eval_every        : int, optional, default: 1
                        Evaluate the training and validation loss every X steps. If we observe no improvement
                        of the validation error, we decrease our patience by 1, else we reset it to *patience*.

    Returns
    -------
    best_Q            : np.array, shape [N, k]
                        Best value for Q (based on validation loss) observed during training
                      
    best_P            : np.array, shape [k, D]
                        Best value for P (based on validation loss) observed during training
                      
    validation_losses : list of floats
                        Validation loss for every evaluation iteration, can be used for plotting the validation
                        loss over time.
                        
    train_losses      : list of floats
                        Training loss for every evaluation iteration, can be used for plotting the training
                        loss over time.                     
    
    converged_after   : int
                        it - patience*eval_every, where it is the iteration in which patience hits 0,
                        or -1 if we hit max_steps before converging. 

    """
    ### YOUR CODE HERE ###
    def training_loss(Q,P,non_zero_idx):
        ''' calculates the training loss'''
        R_mat=np.dot(Q,P)
        true_R=M[non_zero_idx[:,0],non_zero_idx[:,1]].A1
        pred_R=R_mat[non_zero_idx[:,0],non_zero_idx[:,1]]
        SSE_train= np.sum((pred_R-true_R)**2)
        train_losses.append(SSE_train)
        return SSE_train    
    
    def validation_loss(Q,P,val_idx,val_values):
        ''' Calculates the validation loss'''
        loop_index=0
        SSE=0
        for row, col in val_idx:
            true_r=val_values[loop_index]
            loop_index+=1
            pred_r=np.dot(Q[row,:],P[:,col])
            SSE+=(pred_r-true_r)**2 #sum of squared errors 
        validation_losses.append(SSE)
        return SSE
    
    validation_losses=[]
    train_losses=[]
    Q, P = initialize_Q_P(M, k, init)
    reg= reg_lambda * np.eye(k,k)
    N=M.shape[0]
    
    for iteration in range(15):
        tic=time.time()
        
        #update for Q
        for x in range(N):#N
            #searching for all items that have been rated by user x
            idx = non_zero_idx[non_zero_idx[:,0]==x][:,1] # [:,1] to get the col values corresponding to row value == x
            reg = Ridge (alpha = reg_lambda)
            reg.fit(P[:,idx].T,M[x,idx].todense().A1)
            Q[x,]= reg.coef_
        toc=time.time()
        
        #print("to update Q took time", toc-tic)
        
        # update for P      
        D= M.shape[1]
        tic=time.time()
        for j in range(D):
            #searching for all users who have rated item j
            #[:,0] to get the row values corresponding to col value == j
            idx = non_zero_idx[non_zero_idx[:,1]==j][:,0] #searching for all rows indx in the non_zero_idx array that has a col value =j, [0] to get the array from the tuple output of np.where
            reg = Ridge (alpha = reg_lambda)
            reg.fit(Q[idx,],M[idx,j].todense().A1)
            P[:,j]= reg.coef_
        toc=time.time()
    
        #print("to update P took time", toc-tic)    

        #get the values then print it 
        v_loss=validation_loss(Q,P,val_idx,val_values)
        t_loss=training_loss(Q,P,non_zero_idx)
        print("iteration "+ str(iteration)+", " + "training loss: "+ str(t_loss) + ", validation loss: " + str(v_loss))

    
    
    best_Q=Q
    best_P=P
    converged_after=0
    return best_Q, best_P, validation_losses, train_losses, converged_after

tic=time.time()
Q_a, P_a, val_l_a, tr_l_a, conv_a = latent_factor_alternating_optimization(M_shifted, nonzero_indices, 
                                                                           k=100, val_idx=val_idx,
                                                                           val_values=val_values_shifted, 
                                                                           reg_lambda=1, init='random',
                                                                           max_steps=100, patience=10)

toc=time.time()

print("time consumed for the optimization",toc-tic)