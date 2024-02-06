# %%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
def embed_time_series(data, window_size):
    # function to form a hankel matrix from delay-embeddings from time series
    # data                  -   a snippet/observation window from a 1d time series
    # window_size           -   window_size in samples
    N = len(data)
    K = N - window_size+1
    hankel_matrix = np.zeros((window_size, K))
    for i in range(K):
        hankel_matrix[:, i] = data[i:i+window_size]
    return hankel_matrix

# %%
def decompose_hankel_matrix(hankel):
    U, S, Vt = np.linalg.svd(hankel,full_matrices=False)
    return U, S, Vt


# %%
def mean_anti_diagonals(A):
    # Get the dimensions of the matrix A
    rows, cols = A.shape
    #print(rows)
    #print(cols)
    # Initialize a list to store the anti-diagonal means
    anti_diagonal_means = []

    # Calculate the anti-diagonal means
    for k in range(1-rows, cols):
        anti_diagonal = np.diagonal(np.flipud(A), offset=k)
        anti_diagonal_means.append(np.mean(anti_diagonal))
    return np.array(anti_diagonal_means)



# %%
def reconstruct_time_series(U, S, Vt, selected_indices):
    reconstructed_matrix = np.zeros_like(U)
    for i in selected_indices:
        reconstructed_matrix += np.outer(U[:, i], S[i] * Vt[i, :])
    #print(reconstructed_matrix.shape)
    reconstructed_series = mean_anti_diagonals(reconstructed_matrix)
    #print(len(reconstructed_series))
    return reconstructed_series


# %%
# test reconstruct_time_series
#matrix1 = np.array([[1,1,1],[3,2,1],[1,1,1]])
#matrix2 = np.array([[1,3,1],[3,3,3],[1,1,1]])
#reconstruct_time_series(matrix1,np.arange(3), matrix2,[1])


# %%
def get_C_for_component_i(U1, U2,i):
    val = 0
    #print(U2.shape)
    for j in range(U2.shape[1]):
        temp = np.dot( U1[:,i], U2[:,j])
        #print(temp)
        if temp > val :
            val = temp
    return val

# %%
# test get_C_for_component_i
# matrix1 = np.array([[1,1,1],[2,2,2],[3,3,3]])
# matrix2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
# val = get_C_for_component_i(matrix1, matrix2,0)


# %%
def component_selection(Ux, Uy, Uz, tau):
    indices = list()
    for i in range(Ux.shape[1]):
        # here we have to decide which direction we consider "main axis"
        C_xy = get_C_for_component_i(Ux, Uy, i)
        C_xz = get_C_for_component_i(Ux, Uz, i)
        if max(C_xy,C_xz)<tau:
            indices.append(i)
    return np.array(indices)

# %%
def do_ssa_firstn(time_series, lagged_window_size = 1001, first_n_components = 10,):
    # perform singular spectrum analysis, based on singular value decomposition and 
    # thresholding for component selection, and return reconstructed signal that
    # should mainly include the "heart beating signal".
    

    window_x = time_series[:,0]
    window_y = time_series[:,1]
    window_z = time_series[:,2]

    hankel_matrix_x = embed_time_series(window_x-np.mean(window_x), lagged_window_size)
    hankel_matrix_y = embed_time_series(window_y-np.mean(window_y), lagged_window_size)
    hankel_matrix_z = embed_time_series(window_z-np.mean(window_z), lagged_window_size)
    Ux, Sx, Vxt = decompose_hankel_matrix(hankel_matrix_x)
    Uy, Sy, Vyt = decompose_hankel_matrix(hankel_matrix_y)
    Uz, Sz, Vzt = decompose_hankel_matrix(hankel_matrix_z)

    
    # self chosen principal components
    selected_indices = list(range(first_n_components))
    reconstructed_signal_x = reconstruct_time_series(Ux, Sx, Vxt, selected_indices) # here also select primary axis
    reconstructed_signal_y = reconstruct_time_series(Uy, Sy, Vyt, selected_indices)
    reconstructed_signal_z = reconstruct_time_series(Uz, Sz, Vzt, selected_indices)
    reconstructed_signal = np.sqrt(reconstructed_signal_x**2 + reconstructed_signal_y**2 + reconstructed_signal_z**2)
    return reconstructed_signal

def do_ssa_axis(time_series, lagged_window_size = 1001, main_axis = "x", tau=0.6):
    # perform singular spectrum analysis, based on singular value decomposition and 
    # thresholding for component selection, and return reconstructed signal that
    # should mainly include the "heart beating signal".
    

    window_x = time_series[:,0]
    window_y = time_series[:,1]
    window_z = time_series[:,2]

    hankel_matrix_x = embed_time_series(window_x-np.mean(window_x), lagged_window_size)
    hankel_matrix_y = embed_time_series(window_y-np.mean(window_y), lagged_window_size)
    hankel_matrix_z = embed_time_series(window_z-np.mean(window_z), lagged_window_size)
    Ux, Sx, Vxt = decompose_hankel_matrix(hankel_matrix_x)
    Uy, Sy, Vyt = decompose_hankel_matrix(hankel_matrix_y)
    Uz, Sz, Vzt = decompose_hankel_matrix(hankel_matrix_z)

    # condition used in the zhao paper 
    if main_axis == "x":
        selected_indices = component_selection(Ux,Uy,Uz,tau)
        reconstructed_signal = reconstruct_time_series(Ux, Sx, Vxt, selected_indices)
    elif main_axis == "y":
        selected_indices = component_selection(Uy,Ux,Uz,tau)
        reconstructed_signal = reconstruct_time_series(Uy, Sy, Vyt, selected_indices)
    elif main_axis == "z":
        selected_indices = component_selection(Uz,Ux,Uy,tau)
        reconstructed_signal = reconstruct_time_series(Uz, Sz, Vzt, selected_indices)

    return reconstructed_signal

# %%
