
## LDPC function
def generate_ldpc_I(m, k): # generate_ldpc_h 함수에서 사용되는 I 함수를 만들기 위한 함수
    bI = I = np.eye(k, dtype=int)
    I = np.hstack((bI[:,-m:],bI[:,:-m]))

    return I


def generate_ldpc_h(k): # subblock 수에 따라서 0,1 로 된 H matrix 만들기

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

def generate_ldpc_G(n, k, H): # 완성된 H matrix를 가지고 G matrix 만드는 함수 
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
                    E[j,i] = np.log(( 1 + np.prod(M[j,:]) \
                                     / M[j,i]) / ( 1 - np.prod(M[j,:]) / M[j,i]) )
                  #  E[j,i] = np.log((1 + np.prod(M[j,:]) \
                   #                  / M[j,i]+ 1e-10)) / ( 1 - np.prod(M[j,:]) / M[j,i]+ 1e-10))

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

## Base Matrix
P = np.array([[48,29,28,39, 9,61,-1,-1,-1,63,45,80,-1,-1,-1,37,32,22, 1, 0,-1,-1,-1,-1],
            [ 4,49,42,48,11,30,-1,-1,-1,49,17,41,37,15,-1,54,-1,-1,-1, 0, 0,-1,-1,-1],
            [35,76,78,51,37,35,21,-1,17,64,-1,-1,-1,59, 7,-1,-1,32,-1,-1, 0, 0,-1,-1],
            [ 9,65,44, 9,54,56,73,34,42,-1,-1,-1,35,-1,-1,-1,46,39, 0,-1,-1, 0, 0,-1],
            [ 3,62, 7,80,68,26,-1,80,55,-1,36,-1,26,-1, 9,-1,72,-1,-1,-1,-1,-1, 0, 0],
            [26,75,33,21,69,59, 3,38,-1,-1,-1,35,-1,62,36,26,-1,-1, 1,-1,-1,-1,-1, 0]], dtype = int) # coderate =3/4, codeword가 1944d일때 p matrix

## Main
file_path = './sunset.mp4' # MPEG 파일 경로 # 지금은 bch로 받는 codeword 가 없기 때문에 영상 데이터 사용(아래가 영상데이터를 비트로 변환하는 과정)

## Open in Bytes
with open(file_path, 'rb') as mpeg_file:
    b = mpeg_file.read()

## convert it to bits
bits = ''.join(format(byte, '08b') for byte in b)
bits_array = np.array(list(bits), dtype=int) # 영상 데이터를 비트로 변환


## Encoding / Decoding 부분
subblock = 81

H = generate_ldpc_h(subblock) # H matrix 만들기
n = len(H.T)
k = n-len(H)
G = generate_ldpc_G(n, k, H) # G matrix 만들기


fifo_for_encode = 1458 # 데이터 길이 개수
chunks_ldpc = [bits[i:i + fifo_for_encode] for i in range(0, len(bits), fifo_for_encode)] #  데이터 자르기
joined_bits = ''.join(chunks_ldpc)
ldpc_bits_array = np.array(list(joined_bits), dtype=int)


decoded_all = []
for j in range(0, len(ldpc_bits_array, fifo_for_encode): # 1458로 자른 데이터를 encoding하고 decoding 하는 과정 반복
    a = ldpc_bits_array[j:j+fifo_for_encode]
    if(len(a) != fifo_for_encode): # 만약 제일 마지막 chunk가 1458 개로 떨어지지 않으면 나머지 부분 제로패딩
        zeros_to_add = fifo_for_encode-len(a)
        a = np.concatenate((a, np.zeros(zeros_to_add, dtype=int)))

    codeword_chuncks = np.dot(a, G) % 2 # encoding 해서 codeword 만들기 (codeword 길이는 1944)

    original_array = codeword_chuncks
    updated_array = [-1.3863 if value == 1 else 1.3863 for value in original_array] # llr 값 계산을 못해서 일단 임의로 숫자를 넣음
    decoded_data = SPA(H).decode(updated_array) # decoding 

    codeword_all = np.hstack((decoded_all, decoded_data[:1458])) # chunk 별로 decoding 한 값을 하나로 합치기