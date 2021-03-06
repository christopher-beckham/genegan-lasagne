import numpy as np
import h5py
from keras.preprocessing.image import ImageDataGenerator
    
class MnistIterator():
    def _iterator(self,X_Au, X_B0, bs, shuffle):
        while True:
            if shuffle:
                np.random.shuffle(X_Au)
                np.random.shuffle(X_B0)
            for b in range(self.N // bs):
                yield X_Au[b*bs:(b+1)*bs], X_B0[b*bs:(b+1)*bs]
    def __init__(self, mode, c1, c2, bs, shuffle):
        assert mode in ['train', 'valid']
        from load_mnist import load_dataset
        X_train, y_train, X_valid, y_valid, _, _ = load_dataset()
        if mode == 'train':
            X_Au = X_train[y_train == c1]
            X_B0 = X_train[y_train == c2]
        else:
            X_Au = X_valid[y_valid == c1]
            X_B0 = X_valid[y_valid == c2]            
        self.fn = self._iterator(X_Au, X_B0, bs, shuffle)
        self.N = min(X_Au.shape[0], X_B0.shape[0])
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()

def _get_slices(length, bs):
    slices = []
    b = 0
    while True:
        if b*bs >= length:
            break
        slices.append( slice(b*bs, (b+1)*bs) )
        b += 1
    return slices


# this just wraps the above functional iterator
class Hdf5TwoClassIterator():
    def __init__(self, X, y, bs, imgen, c1, c2, tanh_norm=True, rnd_state=np.random.RandomState(0), debug=False):
        """
        :X: in our case, the heightmaps
        :y: in our case, the textures
        :bs: batch size
        :imgen: optional image data generator
        """
        # build the list of indices corresponding to c1, and c2
        self.idxs_c1 = np.where(y[:]==c1)[0] # e.g. 0..10
        self.idxs_c2 = np.where(y[:]==c2)[0] # e.g. 10..90
        if debug:
            print "idxs_c1", idxs_c1, "length =", len(idxs_c1)
            print "idxs_c2", idxs_c2, "length =", len(idxs_c2)
        # save slices
        self.slices_for_c1 = _get_slices(len(self.idxs_c1), bs)
        self.slices_for_c2 = _get_slices(len(self.idxs_c2), bs)
        # book-keeping
        self.N = min(len(self.idxs_c1), len(self.idxs_c2))
        self.bs = bs
        self.X = X
        self.rnd_state = rnd_state
        self.tanh_norm = tanh_norm
        self.imgen = imgen
        self.fn = self._iterate()
    def _iterate(self):
        while True:
            if self.rnd_state != None:
                self.rnd_state.shuffle(self.slices_for_c1)
                self.rnd_state.shuffle(self.slices_for_c2)
            for elem1,elem2 in zip(self.slices_for_c1, self.slices_for_c2):
                this_X, this_Y = self.X[ self.idxs_c1[elem1].tolist() ], self.X[ self.idxs_c2[elem2].tolist() ]
                if this_X.shape[0] != this_Y.shape[0]:
                    # batch size mis-match, go to start of while loop
                    break
                if self.tanh_norm:
                    # go between [0,1], then go to [-1, 1]
                    norm_params = {'axis': (1,2,3), 'keepdims':True}
                    this_X = (this_X - np.min(this_X,**norm_params)) / ( np.max(this_X,**norm_params) - np.min(this_X,**norm_params) )
                    this_X = (this_X - 0.5) / 0.5
                    this_Y = (this_Y - np.min(this_Y,**norm_params)) / ( np.max(this_Y,**norm_params) - np.min(this_Y,**norm_params) )
                    this_Y = (this_Y - 0.5) / 0.5
                    # if we passed an image generator, augment the images
                if self.imgen != None:
                    seed = self.rnd_state.randint(0, 100000)
                    this_X = self.imgen.flow(this_X, None, batch_size=self.bs, seed=seed).next()
                    this_Y = self.imgen.flow(this_Y, None, batch_size=self.bs, seed=seed).next()
                yield this_X, this_Y
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()

if __name__ == '__main__':
    #itr = MnistIterator('train', 0, 9, 32, True)
    #print itr

    dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    dataset = h5py.File(dr_h5,"r")
    imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    it_train = BasicHdf5Iterator(X=dataset['xt'],
                            y=dataset['yt'],
                            bs=100,
                            imgen=imgen,
                            c1=0,
                            c2=4,
                            rnd_state=np.random.RandomState(42),
                            tanh_norm=True,
                            debug=True)
    from skimage.io import imsave
    from util import convert_to_rgb
    grid = np.zeros((256*10,256*10,3)).astype("float32")
    grid2 = np.zeros((256*10,256*10,3)).astype("float32")
    x,y = it_train.next()
    ctr = 0
    for i in range(10):
        for j in range(10):
            grid[i*256:(i+1)*256, j*256:(j+1)*256, :] = convert_to_rgb(x[ctr])
            grid2[i*256:(i+1)*256, j*256:(j+1)*256, :] = convert_to_rgb(y[ctr])            
            ctr += 1
    imsave(arr=grid, fname="grid0.png")
    imsave(arr=grid2, fname="grid4.png")    
    
    #x,y = it_train.next()
    #import pdb
    #pdb.set_trace()
