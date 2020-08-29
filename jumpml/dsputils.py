import numpy as np

def compute_meanSquare(x):
    return np.mean(x**2)

def compute_sumOfSquares(x):
    return np.sum(x**2)

def compute_norm(x):
    return np.sqrt(np.sum(x**2))

def compute_RMS(x):
    return np.sqrt(np.mean(x**2))
    
def find_peak(x):
    return (x.max(), x.argmax())

def update_buffer(x, buffer):
    frameSize = len(x)
    buffer[0:frameSize] = buffer[frameSize:]
    buffer[frameSize:] = x

def apply_expSmoothing(Curr, Prev, gamma):
    return ((1.0 - gamma) * Curr + gamma * Prev)