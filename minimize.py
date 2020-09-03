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
                    df3 = grad(X+x3*s, *args)
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                        print('error')
                    SUCCESS = 1
                except:
                    x3 = (x2+x3)/2
        if f3 <= F0 :
            X0 = X + x3*s
            F0 = f3
            dF0 = df3
        d3 = df3.T *s
        if (d3 > SIG*d0) or (f3 > (f0+x3*RHO*d0)) or M == 0:
            break

        x1 = x2, f1 = f2, d1 = d2
        x2 = x3, f2 = f3, d2 = d3
        A = 6 * (f1-f2) + 3 * (d2+d1) * (x2-x1)
        B = 3 * (f2-f1) - (2*d1+d2) * (x2-x1)
        x3 = x1-d1*(x2-x1)^2/(B+np.sqrt(B*B-A*d1*(x2-x1)))

        if ((not isreal(x3)) or (isnan(x3)) or (isinf(x3)) or x3 ) < 0:
            x3 = x2*EXT
        elif x3 > x2*EXT:
            x3 = x2*EXT
        elif x3 < x2+ INT*(x2-x1):
            x3 = x2+INT*(x2-x1)
        else:
            break

        while ((abs(d3) > -SIG*d0) or (f3 > f0+x3*RHO*d0)) and M > 0:
            if d3 > 0 or f3 > f0+x3*RHO*d0 :                   
                x4 = x3; f4 = f3; d4 = d3          
            else:
                x2 = x3; f2 = f3; d2 = d399
            if f4 > f0:
                x3 = x2 - (0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2))
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)   
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
            f3 = f(X+x3*s, *args) 
            if f3 < F0:
                X0 = X+x3*s
                F0 = f3
                dF0 = df3
            M -= 1 
            i += (length<0)
            d3 = df3.T * s
        
        if (abs(d3) < -SIG*d0) and (f3 < f0+x3*RHO*d0):
            X = X + x3*s
            f0 = f3
            fX = ...#line 138 octave minimize.m
            print('{} {}; Value {}'.format(S, i, f0))
            s = (df3.T * df3-df0.T*df3)/(df0.T*df0)*s - df3
            df0 = df3
            d3 = d0
            d0 = df0.T*s

            if d0 > 0:
                s= -df0
                d0 = -s.T * s
            x3 = x3 * min(RATIO, d3/(d0-realmin))
            is_failed = 0
        else:
            X = X0, f0 = F0, df0 = dF0
            if is_failed or (i > abs(length)):
                break
            s = -df0, d0 = -s.T * s
            x3 = 1/(1-d0)
            is_failed = 1
    print()



    return x3