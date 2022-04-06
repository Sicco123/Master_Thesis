import numpy as np

def hramp_prime(x, lower, eps):
    if len(lower) > 1 :
        return np.stack(np.vectorize(hramp_prime)(x,lower,eps))
    else:
        if lower == float("-inf"):
            return 1
        else:
            dhr = (x-lower)/eps
            dhr[x>(lower+eps)] = 1
            dhr[x<lower] = 0
            return(dhr)

def huber_prime(x, eps):
    dh = x/eps
    dh[x > eps] = 1
    dh[x < -eps] = -1
    dh[np.isnan(dh)] = 0
    return dh

def huber(x, eps):
    h = np.abs(x)-eps/2 if np.abs(x) > eps else (x**2)/(2*eps)
    h[np.isnan(h)] = 0
    return h

def hramp(x, lower, eps):
        if lower == float("-inf"):
            return x
        else:
            return huber(x-lower, eps) + lower if x > lower else 0



