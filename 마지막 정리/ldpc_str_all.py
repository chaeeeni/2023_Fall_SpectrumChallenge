from moviepy.editor import VideoFileClip
import numpy as np

file_path = './sunset.mp4' # MPEG 파일 경로
# file_path
## Open in Bytes
with open(file_path, 'rb') as mpeg_file:
    b = mpeg_file.read()

## convert it to bits
bits = ''.join(format(byte, '08b') for byte in b)


## LDPC function
def generate_ldpc_I(m, k):
    bI = I = np.eye(k, dtype=int)
    I = np.hstack((bI[:,-m:],bI[:,:-m]))

    return I


def generate_ldpc_h(k):

    for i in range(len(P)):
        for j in range(len(P.T)):
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


## LDPC Decoding function
class SPA:
    """ This class can apply SPA algorithm to received LLR vector r.

    Parameters
    ----------
    H: numpy.array
        Parity-Check matrix.
    Imax: int, optional
        Maximum number of iterations.
    trace_on: bool, optional
        To print or not to print intermediate results of calculations.

    Attributes
    ----------
    H: 2D numpy.array
        Parity-Check matrix.
    Imax: int
        Maximum number of iterations.
    trace_on: bool
        To print or not to print intermediate results of calculations.
    H_0: int
        Number of rows of the Parity-Check matrix.
    H_1: int
        Number of columns of the Parity-Check matrix.
    H_mirr: 2D numpy.array:
        'Mirror' of the Parity-Check matrix.
    """

    def __init__(self, H, Imax=1000, trace_on=True):
        self.H = H
        self.Imax = Imax
        self.trace_on = trace_on
        self.H_0 = np.shape(H)[0]
        self.H_1 = np.shape(H)[1]
        self.H_mirr = (self.H + np.ones(np.shape(self.H))) %2

    def __nrz(self, l):
        """Applies inverse NRZ

        Parameters
        ----------
        l: 1D numpy.array
            LLR vector.

        Returns
        -------
        l: 1D numpy.array
            Mapped to binary symbols input vector.
        """

        for idx, l_j in enumerate(l):
            if l_j >= 0:
                l[idx] = 0
            else:
                l[idx] = 1
        return l

    def __calc_E(self, E, M):
        """ Calculates V2C message

        Parameters
        ----------
        E: 2D numpy.array
            Current V2C matrix.
        M: 2D numpy.array
            Current C2V matrix.

        Returns
        -------
        E: 2D numpy.array
            Updated V2C matrix.
        """

        M = np.tanh(M / 2) + self.H_mirr
        for j in range(self.H_0):
            for i in range(self.H_1):
                if self.H[j,i] != 0:
                    #E[j,i] = np.log((1 + np.prod(M[j,:]) \
                     #                / M[j,i]+ 1e-10)) / ( 1 - np.prod(M[j,:]) / M[j,i]+ 1e-10)))
                    # E[j, i] = np.log((1 + np.prod(M[j, :]) / (M[j, i] + 1e-10)) / (1 - np.prod(M[j, :]) / (M[j, i] + 1e-10)))
                    if M[j, i] > 0:
                        E[j, i] = np.log((1 + np.prod(M[j, :]) / (M[j, i] + 1e-10)) / (1 - np.prod(M[j , :]) / (M[j, i] + 1e-10)))
                    else:
                        E[j, i] = 0

        return E

    def __calc_M(self, M, E, r):
        """ Calculates C2V message

        Parameters
        ----------
        M: 2D numpy.array
            Current C2V matrix.
        E: 2D numpy.array
            Current V2C matrix.
        r: 1D numpy.array
            Input LLR vector.

        Returns
        -------
        M: 2D numpy.array
            Updated C2V matrix.
        """

        for j in range(self.H_0):
            for i in range(self.H_1):
                if self.H[j,i] != 0:
                    M[j,i] = np.sum(E[:, i]) - E[j,i] + r[i]
        M = M*H
        return M

    def decode(self, r):

        """Applies SPA algorithm to received LLR vector r.

        Parameters
        ----------
        r: numpy.array of floats
            received from demodulator LLR vector.

        Returns
        -------
        l: numpy.array
            Decoded message.

        """
        stop = False # stopping flag
        I = 0 # maximum number of iterations
        M = np.zeros(np.shape(H)) # C2V
        E = np.zeros(np.shape(H)) # V2C
        l = np.zeros(np.shape(r)) # LLR vector -> decoded message
        #print('H:\n'+str(H))

        while stop == False and I != self.Imax:

            #""" 1) Initial step """
            if I == 0:
                for j in range(np.shape(H)[0]):
                    M[j, :] = r*H[j, :]
                    #print("M")
                    #print(M)
            # if self.trace_on == True:
            #     print('M:\n')

            #""" 2) V2C step """
            E = self.__calc_E(E, M)
            #print("E")
            #print(E)
            # if self.trace_on == True:
            #     print('E:\n')

            #""" 3) Decoded LLR vector """
            l = r + np.sum(E, axis=0)
            # if self.trace_on == True:
            #     print('l:\n')

            #""" 4) NRZ mapping """
            l = self.__nrz(l)
            # if self.trace_on == True:
            #     print('decoded:\n')

            #""" 5) Syndrom checking """
            s = np.dot(H, l) %2
            if np.prod(s == np.zeros(np.size(s))) == 1:
                stop = True
            else:
                I = I + 1
                M = self.__calc_M(M, E, r)
        return l

## ENCODING
subblock = 81

H = generate_ldpc_h(subblock)

n = len(H.T)
k = n-len(H)

G = generate_ldpc_G(n, k, H)


a = insert_shortening
codeword_final = np.dot(a, G) % 2

original_array = codeword_final
updated_array = [-1.3863 if value == 0 else 1.3863 for value in original_array]


##
l2 = SPA(H).decode(updated_array)

##
original_array2 = l
a = l

##
updated_array2 = [0 if value == 1 else 1 for value in original_array2]

##
np.array_equal(rx_data_f[:1458],l[:1458])


## FIFO
fifo = 2000
chunks = [bits[i:i + fifo] for i in range(0, len(bits), fifo)]
joined_bits = ''.join(chunks)

## convert bits to bytes
padded_bit_string = joined_bits.ljust((len(joined_bits) + 7) // 8 * 8, '0')
byte_data = bytes([int(padded_bit_string[i:i+8], 2) for i in range(0, len(padded_bit_string), 8)])
#byte_data = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])


## bytes to mp4
new_file_path = './sunset_new.mp4'
with open(new_file_path, 'wb') as file:
        file.write(byte_data)

## play video
clip = VideoFileClip(new_file_path)
clip.preview()
clip.close()