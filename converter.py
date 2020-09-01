import array
import numpy as np
import scipy.io as sio
from cell_matlab import cell
# from astropy.io import ascii

print('You first need to download files:')
print('train-images-idx3-ubyte.gz')
print('train-labels-idx1-ubyte.gz')
print('t10k-images-idx3-ubyte.gz')
print('t10k-labels-idx1-ubyte.gz')
print('from http://yann.lecun.com/exdb/mnist/')
print('and gunzip them')

def converter():
    while True:
        try:
            with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-images.idx3-ubyte", 'rb') as f:
                A= np.fromfile(f, np.int32).reshape((-1, 4))
                print(A.shape)
            break
        except:
            print("There's no file named it")
    
    with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-labels.idx1-ubyte", 'rb') as g:
        l = np.fromfile(g, np.int32).reshape((-1, 2))
        print(l.shape)

    print('Starting to convert Test MNIST images (prints 10 dots)')
    N = 1000

    Df = cell(1,10)
    print('Df :',Df)
    print('Df[1] :',Df[1])
    for d in range(10):
        print(d)
        print(Df[d])
        Df[d].append(open('test'+ str(d) +'.ascii', 'w'))
        print("New Df[d] =", Df[d])
    Z =np.concatenate( Df, axis=0 )
    print("Df after append =", Df)
    print(A.shape)
    for i in range(1, 10):
        print('.', end = '')
        import struct
        with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-images.idx3-ubyte", 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            raw_images = np.fromfile(f, dtype=np.uint8).reshape(rows * cols, num)  # uint8
            raw_images = raw_images.astype(np.float32) / 255.0
        with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-labels.idx1-ubyte", 'rb') as g:
            num = struct.unpack(">II", g.read(8))
            raw_labels = np.fromfile(g, dtype='uint8')
        print("raw_labels.shape =",str(raw_labels.shape))
        print("raw_images.shape",str(raw_images.shape))
        for j in range(1, N):
            print(j)
            Df[raw_labels[j]][0].write(str(raw_images[:j]))
            print(Df[raw_labels[j]][0])
            print("--------------------------------------")
    print()
    print("Df", Z[1])
    # S = ascii.read('test'+str(1)+'.ascii')
    # print('Digits of class', S['D'].shape)
    for d in range(9):
        Z[d].close()
        D = sio.loadmat('test'+str(d)+'.ascii')
        print('Digits of class', D['D'].shape)
        sio.savemat('test' + str(d)+'.mat', {'vect':D['D']})
        
    with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//train-images.idx3-ubyte", 'rb') as f:
        a = np.fromfile(f, np.int32).reshape((-1, 4)).T
        print(a)
    with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//train-labels.idx3-ubyte", 'rb') as f:
        l = np.fromfile(f, np.int32).reshape((-1, 2)).T
        print(l)
    print('Starting to convert Training MNIST images (prints 60 dots)')
    N = 1000

    Df = np.empty(shape=(1,10),dtype='object')

    for d in range(9):
        np.append(Df[0][d], open('digit'+ str(d) +'.mat', 'w'))

    for i in range(1, 60):
        print('.')
        with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-images.idx3-ubyte", 'rb') as f:
            raw_images = np.fromfile(f, np.ubyte)
        with open("C:/Users/LENOVO/skripsi/Autoencoder_Code//t10k-labels.idx1-ubyte", 'rb') as g:
            raw_labels = np.fromfile(g, np.ubyte)
        for j in range(1, N):
            print(Df[0][rawlabels(j)+1],rawimages[:j])
            print(Df[rawlabels(j)+1])

    print()
    for d in range(9):
        Df[d].close()
        D = sio.loadmat('digit'+str(d)+'.mat')
        print('Digits of class', D['D'].shape)
        sio.savemat('digit' + str(d)+'.mat', {'vect':D['D']})

if __name__ == "__main__":
    converter()
