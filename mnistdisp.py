def mnistdisp(digits):
    import numpy as np
    import matplotlib as plt

    COL = 28, ROW = 28
    dd, N = digits.shape
    IMDISP = np.zeros((2*28, np.ceil(N/2)*28))

    for nn in range(1, N):
        ii = rem(nn, 2)
        if (ii==0):
            ii=2
        jj = np.ceil(nn/2)

        img1 = np.reshape(digits[:,nn], row, col)
        img2(((ii-1)*row+1):(ii*row), ((jj-1)*col+1):(jj*col))= img1.T
    
    """
    imagesc(img2,[0 1]); colormap gray; axis equal; axis off;
    drawnow;
    """
    err = 0

    return err