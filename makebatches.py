def makebatches():
    import scipy.io as sio
    import numpy as np
    import tensorflow

    digit0 = sio.loadmat("digit0.mat")
    digit1 = sio.loadmat("digit1.mat")
    digit2 = sio.loadmat("digit2.mat")
    digit3 = sio.loadmat("digit3.mat")
    digit4 = sio.loadmat("digit4.mat")
    digit5 = sio.loadmat("digit5.mat")
    digit6 = sio.loadmat("digit6.mat")
    digit7 = sio.loadmat("digit7.mat")
    digit8 = sio.loadmat("digit8.mat")
    digit9 = sio.loadmat("digit9.mat")

    #digit0
    DIGIT_DATA = digit0['D']
    print("Shape of DIGIT_DATA = ", DIGIT_DATA.shape)
    TARGETS = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (DIGIT_DATA.shape[0], 1))
    print("Shape of TARGETS= ", TARGETS.shape)

    #digit1
    print("Shape of digit1['D']=", digit1['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit1['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), (digit1['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit2
    print("Shape of digit2['D']=", digit2['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit2['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), (digit2['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit3
    print("Shape of digit3['D']=", digit3['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit3['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), (digit3['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit4
    print("Shape of digit4['D']=", digit4['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit4['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), (digit4['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit5
    print("Shape of digit5['D']=", digit5['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit5['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), (digit5['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit6
    print("Shape of digit6['D']=", digit6['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit6['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), (digit6['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit7
    print("Shape of digit7['D']=", digit7['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit7['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), (digit7['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit8
    print("Shape of digit8['D']=", digit8['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit8['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), (digit8['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #digit9
    print("Shape of digit9['D']=", digit9['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, digit9['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), (digit9['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)

    DIGIT_DATA = DIGIT_DATA/255

    TOTNUM = DIGIT_DATA.shape[0]
    print('Size of the training dataset= ', TOTNUM)

    np.random.rand(0)
    RANDOMORDER = np.random.permutation(range(TOTNUM))
    print(RANDOMORDER.shape)
    NUM_BATCHES = int(round(TOTNUM/100))
    NUM_DIMS = DIGIT_DATA.shape[1]
    BATCH_SIZE = 100
    BATCH_DATA = np.zeros((BATCH_SIZE, NUM_DIMS, NUM_BATCHES))
    BATCH_TARGETS = np.zeros((BATCH_SIZE, 10, NUM_BATCHES))

    for b in range(1, NUM_BATCHES):
        BATCH_DATA[:,:,b] = DIGIT_DATA[RANDOMORDER[(b-1)*BATCH_SIZE:b*BATCH_SIZE],:]
        BATCH_TARGETS[:,:,b] = TARGETS[RANDOMORDER[(b-1)*BATCH_SIZE:b*BATCH_SIZE],:]

    del DIGIT_DATA, TARGETS

    test0 = sio.loadmat("test0.mat")
    test1 = sio.loadmat("test1.mat")
    test2 = sio.loadmat("test2.mat")
    test3 = sio.loadmat("test3.mat")
    test4 = sio.loadmat("test4.mat")
    test5 = sio.loadmat("test5.mat")
    test6 = sio.loadmat("test6.mat")
    test7 = sio.loadmat("test7.mat")
    test8 = sio.loadmat("test8.mat")
    test9 = sio.loadmat("test9.mat")

    #test0
    DIGIT_DATA = test0['D']
    print("Shape of DIGIT_DATA = ", DIGIT_DATA.shape)
    TARGETS = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (DIGIT_DATA.shape[0], 1))
    print("Shape of TARGETS= ", TARGETS.shape)

    #test1
    print("Shape of test1['D']=", test1['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test1['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), (test1['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test2
    print("Shape of test2['D']=", test2['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test2['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), (test2['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test3
    print("Shape of test3['D']=", test3['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test3['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), (test3['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test4
    print("Shape of test4['D']=", test4['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test4['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), (test4['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test5
    print("Shape of test5['D']=", test5['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test5['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), (test5['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test6
    print("Shape of test6['D']=", test6['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test6['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), (test6['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test7
    print("Shape of test7['D']=", test7['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test7['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), (test7['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test8
    print("Shape of test8['D']=", test8['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test8['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), (test8['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)
    #test9
    print("Shape of test9['D']=", test9['D'].shape )
    DIGIT_DATA = np.concatenate((DIGIT_DATA, test9['D']))
    print("Shape of DIGIT_DATA after append=", DIGIT_DATA.shape )
    TARGETS = np.concatenate((TARGETS, np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), (test9['D'].shape[0], 1))))
    print("Shape of TARGETS= ", TARGETS.shape)

    DIGIT_DATA = DIGIT_DATA/255

    TOTNUM = DIGIT_DATA.shape[0]
    print('Size of the testing dataset= ', TOTNUM)

    np.random.rand(0)
    RANDOMORDER = np.random.permutation(range(TOTNUM))
    print(RANDOMORDER.shape)
    NUM_BATCHES = int(round(TOTNUM/100))
    NUM_DIMS = DIGIT_DATA.shape[1]
    BATCH_SIZE = 100
    TEST_BATCH_DATA = np.zeros((BATCH_SIZE, NUM_DIMS, NUM_BATCHES))
    TEST_BATCH_TARGETS = np.zeros((BATCH_SIZE, 10, NUM_BATCHES))

    for b in range(1, NUM_BATCHES):
        TEST_BATCH_DATA[:,:,b] = DIGIT_DATA[RANDOMORDER[(b-1)*BATCH_SIZE:b*BATCH_SIZE],:]
        TEST_BATCH_TARGETS[:,:,b] = TARGETS[RANDOMORDER[(b-1)*BATCH_SIZE:b*BATCH_SIZE],:]

    del DIGIT_DATA, TARGETS

    np.random.seed()
    return BATCH_DATA

if __name__ == "__main__":
    makebatches()