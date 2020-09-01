def backprop(BATCHDATA, VISHID, HIDRECBIASES, HIDPEN, PENRECBIASES):
    import numpy as np
    import scipy.io as sio
    from makebatches import *

    MAX_EPOCH = 200
    print('Fine-tuning deep autoencoder by minimizing cross entropy error.')
    print('60 batches of 1000 cases each.')

    '''
    load mnistvh
    load mnisthp
    load mnisthp2
    load mnistpo 
    '''

    BATCH_DATA = makebatches()
    NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape
    N = NUM_CASES

    w1 = np.append(VISHID, HIDRECBIASES, axis = 0)
    w2 = np.append(HIDPEN, PENRECBIASES, axis = 0)
    w3 = np.append(HIDPEN2, PENRECBIASES, axis = 0)
    w4 = np.append(HIDTOP, TOPRECBIASES, axis = 0)
    w5 = np.append(HIDTOP.T, TOPGENBIASES, axis = 0)
    w6 = np.append(HIDPEN2, HIDGENBIASES, axis = 0)
    w7 = np.append(HIDPEN.T, HIDGENBIASES, axis = 0)
    w8 = np.append(VISHID.T, VISBIASES, axis = 0)

    L1 = W1.shape-1
    L2 = W2.shape-1
    L3 = W3.shape-1
    L4 = W4.shape-1
    L5 = W5.shape-1
    L6 = W6.shape-1
    L7 = W7.shape-1
    L8 = W8.shape-1
    L9 = L1

    test_err=[]
    train_err=[]


    for epoch in range(1, MAX_EPOCH):
        ERR = 0
        NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape
        N = NUM_CASES
        for batch in range(1, NUM_BATCHES):
            data = BATCH_DATA[:,:,batch]
            data = np.ones(N)
            
            W1_PROBS = 1.0/(1 + np.exp(-data * w1))
            W1_PROBS = np.linspace(W1_PROBS, np.ones(N))
            W2_PROBS = 1.0/(1 + np.exp(-W1_PROBS * w2))
            W2_PROBS = np.linspace(W2_PROBS, np.ones(N))
            W3_PROBS = 1.0/(1 + np.exp(-W2_PROBS * w3))
            W3_PROBS = np.linspace(W3_PROBS, np.ones(N))
            W4_PROBS = W3_PROBS * W4
            W4_PROBS = np.linspace(W4_PROBS, np.ones(N))
            W5_PROBS = 1.0/(1 + np.exp(-W4_PROBS * w5))
            W5_PROBS = np.linspace(W5_PROBS, np.ones(N))
            W6_PROBS = 1.0/(1 + np.exp(-W5_PROBS * w6))
            W6_PROBS = np.linspace(W6_PROBS, np.ones(N))
            W7_PROBS = 1.0/(1 + np.exp(-W6_PROBS * w7))
            W7_PROBS = np.linspace(W7_PROBS, np.ones(N))
            DATAOUT = 1.0/(1 + np.exp(-W7_PROBS * w8))
            ERR += 1/N*sum(np.square(sum((data[:,1:len(data)-1]-DATAOUT))))
        TRAIN_ERR = ERR/NUM_BATCHES

    print('Displaying in figure 1: Top row - real data, Bottom row -- reconstructions')
    OUTPUT = []
    for ii in range(1, 15):
        # OUTPUT = 
        '''
        if epoch==1 
        close all 
        figure('Position',[100,600,1000,200]);
        else 
        figure(1)
        end 
        mnistdisp(output);
        drawnow;
        '''
    TESTNUMCASES, TESTNUMDIMS, TESTNUMBATCHES = TESTBATCHDATA.shape
    N = TESTNUMCASES
    ERR = 0
    for batch in range(1, TESTNUMBATCHES):
        data = BATCH_DATA[:,:,batch]
        data = np.ones(N)
        W1_PROBS = 1.0/(1 + np.exp(-data * w1))
        W1_PROBS = np.linspace(W1_PROBS, np.ones(N))
        W2_PROBS = 1.0/(1 + np.exp(-W1_PROBS * w2))
        W2_PROBS = np.linspace(W2_PROBS, np.ones(N))
        W3_PROBS = 1.0/(1 + np.exp(-W2_PROBS * w3))
        W3_PROBS = np.linspace(W3_PROBS, np.ones(N))
        W4_PROBS = W3_PROBS * W4
        W4_PROBS = np.linspace(W4_PROBS, np.ones(N))
        W5_PROBS = 1.0/(1 + np.exp(-W4_PROBS * w5))
        W5_PROBS = np.linspace(W5_PROBS, np.ones(N))
        W6_PROBS = 1.0/(1 + np.exp(-W5_PROBS * w6))
        W6_PROBS = np.linspace(W6_PROBS, np.ones(N))
        W7_PROBS = 1.0/(1 + np.exp(-W6_PROBS * w7))
        W7_PROBS = np.linspace(W7_PROBS, np.ones(N))
        DATAOUT = 1.0/(1 + np.exp(-W7_PROBS * w8))
        ERR += 1/N*sum(np.square(sum((data[:,1:len(data)-1]-DATAOUT))))
    TRAIN_ERR = ERR/NUM_BATCHES
    print('Before epoch {} Train squared error: {} Test squared error: {}'.format(epoch,train_err(epoch),test_err(epoch)))

    TT = 0
    for batch in range(1, NUM_BATCHES/10):
        print('epoch {} batch {}'.format(epoch,batch))

        TT+=1
        data=[]
        for kk in range(1, 10):
            data = 1
        MAX_ITER = 3
        

