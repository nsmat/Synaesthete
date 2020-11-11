import numpy as np

class FourierTransformer():
    """ Performs Fourier Transforms"""
    # TODO Check the calculations Zac did.

    def __init__(self, chunk_size):
        # TODO rename variables to be more meaningful
        self.chunk_size = chunk_size
        self.T = 1.0 / (12*chunk_size)
        self.frequencies = np.linspace(0.0, 1.0/(2*self.T), int(chunk_size/2))

    def transform(self, data):
        yf = np.abs(np.fft.fft(data))
        return  2.0/self.chunk_size *np.abs(yf[:self.chunk_size//2]) # TODO Why stop halfway??
        