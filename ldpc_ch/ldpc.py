# I1 = np.hstack((I[:,-1:],I[:,:-1]))2

import numpy as np
import matplotlib.pylab as plt
import scipy.io


P = np.array([[16,17,22,24, 9, 3,14,-1, 4, 2, 7,-1,26,-1, 2,-1,21,-1, 1, 0,-1,-1,-1 -1],
            [25,12,12, 3, 3,26, 6,21,-1,15,22,-1,15,-1, 4,-1,-1,16,-1, 0, 0,-1,-1,-1],
            [25,18,26,16,22,23, 9,-1, 0,-1, 4,-1, 4,-1, 8,23,11,-1,-1,-1, 0, 0,-1,-1],
            [ 9, 7, 0, 1,17,-1,-1, 7, 3,-1, 3,23,-1,16,-1,-1,21,-1, 0,-1,-1, 0, 0,-1],
            [24, 5,26, 7, 1,-1,-1,15,24,15,-1, 8,-1,13,-1,13,-1,11,-1,-1,-1,-1, 0, 0],
            [ 2, 2,19,14,24, 1,15,19,-1,21,-1, 2,-1,24,-1, 3,-1, 2, 1,-1,-1,-1,-1, 0]])


def generate_ldpc_I(m, k):
    bI = I = np.eye(k, dtype=int)
    I = np.hstack((bI[:,-m:],bI[:,:-m]))

    return I


def generate_ldpc_h(k):

    for i in range(len(P)):
        for j in range(len(P.T)):
            # print("I:")
            # print(P[i, j])
            if(P[i, j] > 0):
                I = generate_ldpc_I(P[i, j], k)
            elif (P[i, j] == 0):
                I = np.eye(k,dtype=int)
            else :
                I = np.zeros((k, k), dtype=int)


            if(j==0):
                Hj = I
            else:
                Hj = np.hstack((Hj, I))
        if(i==0):
            H = Hj
        else:
            H = np.vstack((H, Hj))

    return H

def generate_ldpc_G(n, k, H):
    G = np.eye(k, dtype=int)
    p = H[:k, :k];
    G = np.hstack((G, p.T))

    return G

##
ext = 27

H = generate_ldpc_h(ext)

n = len(H.T)
k = n-len(H)

G = generate_ldpc_G(n, k, H)


mat_file_name = "data.mat"
databits = scipy.io.loadmat(mat_file_name)
databits = np.ones((1,486))

codeword = np.dot(databits, G) % 2
print(codeword)


