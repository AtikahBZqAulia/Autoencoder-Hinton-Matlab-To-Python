def rbm(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH):

    import numpy as np

    EPSILON_W = 0.1
    EPSILON_VB = 0.1
    EPSILON_HB = 0.1
    WEIGHT_COST = 0.0002
    INITIAL_MOMENTUM = 0.5
    FINAL_MOMENTUM = 0.9

    print(BATCH_DATA.shape)
    NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape

    if RESTART==1:
        RESTART = 0
        EPOCH = 1

        VISHID = 0.1 * np.random.randn(NUM_DIMS, NUM_HID)
        HIDBIASES = np.zeros(NUM_HID)
        VISBIASES = np.zeros(NUM_DIMS)
        print("NUM_CASES =", NUM_CASES)
        print("HIDBIASES =", NUM_HID)
        POSHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        NEGHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        POSPRODS = np.zeros((NUM_DIMS, NUM_HID))
        NEGPRODS = np.zeros((NUM_DIMS, NUM_HID))
        VISHIDINC = np.zeros((NUM_DIMS, NUM_HID))
        HIDBIASINC = np.zeros(NUM_HID)
        VISBIASINC = np.zeros(NUM_DIMS)
        BATCHPOSHIDPROBS = np.zeros((NUM_CASES, NUM_HID, NUM_BATCHES))

    for epoch in range(EPOCH, MAX_EPOCH):
        print('epoch {}'.format(epoch))
        ERR_SUM = 0
        for batch in range(1, NUM_BATCHES):
            print('epoch {} batch {}'.format(epoch,batch))
            
            data = BATCH_DATA[:,:,batch]
            MULT  = np.matmul(-data,VISHID)
            KURANG = np.tile(HIDBIASES, (NUM_CASES, 1))
            POSHIDPROBS = 1.0/(1 + (np.exp(MULT- KURANG)))
            BATCHPOSHIDPROBS[:,:,batch] = POSHIDPROBS
            # print("BATCHPOSHIDPROBS.shape= ", BATCHPOSHIDPROBS.shape)
            POSPRODS = np.matmul(data.T,POSHIDPROBS)
            POSHIDACT = sum(POSHIDPROBS)
            POSVISACT = sum(data)

            POSHIDSTATES = POSHIDPROBS > np.random.rand(NUM_CASES, NUM_HID)
            NEGDATA = 1.0/(1 + np.exp(np.matmul(~POSHIDSTATES, VISHID.T) - np.tile(VISBIASES, (NUM_CASES, 1))))
            NEGHIDPROBS = 1.0/(1 + np.exp(np.matmul(-NEGDATA, VISHID) - np.tile(HIDBIASES, (NUM_CASES,1))))
            NEGPRODS = np.matmul(NEGDATA.T, NEGHIDPROBS)
            NEGHIDACT = sum(NEGHIDPROBS)
            NEGVISACT = sum(NEGDATA)

            ERR = sum(sum(np.square(data-NEGDATA)))
            ERR_SUM += ERR

            if epoch>5:
                MOMENTUM = FINAL_MOMENTUM
            else:
                MOMENTUM = INITIAL_MOMENTUM

            VISHIDINC = MOMENTUM* VISHIDINC + EPSILON_W  * ((POSPRODS-NEGPRODS)/NUM_CASES - WEIGHT_COST*VISHID)
            VISBIASINC = MOMENTUM* VISBIASINC + (EPSILON_VB/NUM_CASES) *(POSVISACT-NEGVISACT)
            HIDBIASINC = MOMENTUM * HIDBIASINC + (EPSILON_HB/NUM_CASES)* (POSHIDACT-NEGHIDACT)
            VISHID += VISHIDINC
            VISBIASES += VISBIASINC
            HIDBIASES += HIDBIASINC
        print('epoch {} error {}'.format(epoch, ERR_SUM))
    return BATCHPOSHIDPROBS, HIDBIASES, VISHID, HIDBIASES, VISBIASES

