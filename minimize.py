from math import *
from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf

def minimize(X, f, grad, args, maxnumlinesearch=None, maxnumfuneval=None, red=1.0, verbose=True):
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG/2

    SMALL = 10.**-16

    if size(MAX.shape, axis=1)==2:
        red=size(2, axis=1)
        length = size(1, axis=1)
    else:
        red = 1

    if length>=0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    i = 0
    is_failed = 0
    f0 = f(X, *args) 
    fX = [f0]
    df0 = grad(X, *args)
    i+= (length<0)
    s = -df0; d0= -dot(s,s)
    x3 = red/(1-d0)

    while i< abs(length):
        i+= (length>0)
        X0 = X; F0 = f0; dF0 = df0

        if length>0:
            M = MAX
        else:
            M = min(MAX, -length-1)

        while 1 :
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0

            SUCCESS = 0
            while (not SUCCESS) and (M > 0):
                try:
                    M = M - 1
                    i+= (length<0)
                    f3 = f(X+x3*s, *args)
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                        print('error')
                        