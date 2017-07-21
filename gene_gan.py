import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from keras.preprocessing.image import ImageDataGenerator
import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import nolearn
#from keras_ports import ReduceLROnPlateau
import pickle
import sys
import gzip

#from util import iterate_hdf5, Hdf5Iterator, convert_to_rgb, compose_imgs, plot_grid

class GeneGAN():
    def print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer.__class__.__name__, layer.output_shape, \
            "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# params:", count_params(l_out)
    def __init__(self,
                 generator_fn, generator_params, 
                 discriminator_fn, discriminator_params,
                 im_shp=(1,28,28),
                 lambda_recon=1.,
                 opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 reconstruction='l1', lsgan=True, verbose=True):
        self.im_shp = im_shp
        self.verbose = verbose
        # get the networks for the p2p network
        dd = generator_fn(**generator_params)
        self.latent_dim = dd['l_feat'].output_shape[1::]
        if verbose:
            self.print_network(dd['out'])
        Au = T.tensor4('Au')
        B0 = T.tensor4('B0')
        a = T.tensor4('a') # let the user input their own latent vector
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = binary_crossentropy
        # generative stuff for Au
        A_for_Au, u_for_Au, decode_into_Au = get_output([dd['l_enc'], dd['l_feat'], dd['out']], Au)
        # **det stuff**
        A_for_Au_det, u_for_Au_det = get_output([dd['l_enc'], dd['l_feat']], Au, deterministic=True)
        decode_into_A0 = get_output(
            dd['out'],
            {dd['l_in']: Au, dd['l_feat']: T.zeros_like(u_for_Au)}
        ) # X_{Au} --> [A,u] --> [A,0] --> X_{A0}
        decode_into_A0_det = get_output(
            dd['out'],
            {dd['l_in']: Au, dd['l_feat']: T.zeros_like(u_for_Au)}, deterministic=True
        ) # X_{Au} --> [A,u] --> [A,0] --> X_{A0} (DET)
        decode_into_Au_using_a = get_output(
            dd['out'], 
            {dd['l_in']: Au, dd['l_feat']: a}
        ) # X_{Au} --> [A,u] --> inject own u' --> [A,u'] --> X_{Au'}
        Au_recon_loss = T.abs_(decode_into_Au - Au).mean()
        # discriminator stuff for Au
        disc = discriminator_fn(**discriminator_params)
        if verbose:
            self.print_network(disc)
        disc_for_A0 = get_output(disc, decode_into_A0)
        disc_for_B0 = get_output(disc, B0)
        # the discriminator returns the probability the image
        # is a ground truth, i.e., it is B0, and not the predicted A0
        disc_1_loss = adv_loss(disc_for_B0, 1.).mean() + adv_loss(disc_for_A0, 0.).mean()
        A0_generator_loss = adv_loss(disc_for_A0, 1.).mean()
        # **Total loss: reconstruction error + GAN loss to distinguish A0 (fake) and B0 (real)**
        total_generator_loss = A0_generator_loss + lambda_recon*Au_recon_loss
        # generative stuff for B0
        B_for_B0, eps_for_B0 = get_output([dd['l_enc'], dd['l_feat']], {dd['l_in']: B0})
        decode_into_Bu = get_output(
            dd['out'],
            {dd['l_in']: B0, dd['l_feat']: u_for_Au}
        ) # X_{B0} --> [B,0] --> inject u from X_{Au} --> [B,u] --> X_{Bu}
        decode_into_B0 = get_output(
            dd['out'], 
            {dd['l_in']: B0, dd['l_feat']: T.zeros_like(u_for_Au)}
        )
        eps_loss = T.abs_(eps_for_B0).mean()
        B0_recon_loss = T.abs_(decode_into_B0 - B0).mean()
        disc2 = discriminator_fn(**discriminator_params)
        disc_for_Au = get_output(disc2, decode_into_Au)
        disc_for_Bu = get_output(disc2, decode_into_Bu)
        # the discriminator returns the probability the image
        # is a ground truth, i.e., it is Au, and not the predicted Bu
        disc_2_loss = adv_loss(disc_for_Au, 1.).mean() + adv_loss(disc_for_Bu, 0.).mean()
        Bu_generator_loss = adv_loss(disc_for_Bu, 1.).mean()
        # **Total loss: reconstruction error + GAN loss to distinguish Bu (fake) and Au (real)**
        total_generator_loss_2 = Bu_generator_loss + lambda_recon*B0_recon_loss + eps_loss
        # parallelogram loss
        #pgram_loss = T.abs_(Au + B0 - decode_into_A0 - decode_into_Bu).mean()*0.01
        # TOTAL GENERATOR LOSS
        total_loss = total_generator_loss + total_generator_loss_2
        # -----------------------------
        tot_gen_params = get_all_params(dd['out'], trainable=True)
        disc1_params = get_all_params(disc, trainable=True)
        disc2_params = get_all_params(disc2, trainable=True)
        gen_updates = opt(total_loss, tot_gen_params, **opt_args)
        gen_updates.update(opt(disc_1_loss, disc1_params, **opt_args))
        gen_updates.update(opt(disc_2_loss, disc2_params, **opt_args))
        keys = [A0_generator_loss, Bu_generator_loss, Au_recon_loss, B0_recon_loss, eps_loss, total_loss, disc_1_loss, disc_2_loss]
        self.train_fn = theano.function([Au, B0], keys, updates=gen_updates)
        self.loss_fn = theano.function([Au, B0], keys)
        # decompose Au into [A,u]
        # of course it's notational -- you can also apply this to
        # B0 to get [B,0]
        self.enc_fn = theano.function([Au], [A_for_Au, u_for_Au])
        self.enc_fn_det = theano.function([Au], [A_for_Au_det, u_for_Au_det])
        # TODO: custom dec function??
        self.dec_fn = None
        # decompose Au into [A,0] then automatically decode
        self.zero_fn = theano.function([Au], decode_into_A0)
        self.zero_fn_det = theano.function([Au], decode_into_A0_det)
        # decompose Au into [A,0], replace 0 with a (input into the function),
        # then decode
        self.dec_use_a_fn = theano.function([Au, a], decode_into_Au_using_a)
        # ------------
        self.lr = opt_args['learning_rate']
        self.train_keys = ['A0_gen', 'Bu_gen', 'Au_recon', 'B0_recon', 'eps', 'gen_tot', 'B0_A0_disc', 'Au_Bu_disc'] # TODO:
        self.dd = dd
        #self.dd_dec = dd_dec
        self.disc = disc
        self.disc2 = disc2
    def save_model(self, filename):
        with gzip.open(filename, "wb") as g:
            pickle.dump({
                'gen': get_all_param_values(self.dd['out']),
                'disc': get_all_param_values(self.disc),
                'disc2':get_all_params(self.disc2)
            }, g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        """
        filename:
        mode: what weights should we load? E.g. `both` = load
          weights for both p2p and dcgan.
        """
        with gzip.open(filename) as g:
            dd = pickle.load(g)
            set_all_param_values(self.dd['out'], dd['gen'])
            set_all_param_values(self.disc['disc'], dd['disc'])
            set_all_param_values(self.disc2['disc2'], dd['disc2'])
            
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=10, resume=False, reduce_on_plateau=False, schedule={}, quick_run=False):
        def _loop(fn, itr):
            rec = [ [] for i in range(len(self.train_keys)) ]
            for b in range(itr.N // batch_size):
                A_batch, B_batch = it_train.next()
                results = fn(A_batch,B_batch)
                for i in range(len(results)):
                    rec[i].append(results[i])
                if quick_run:
                    break
            return tuple( [ np.mean(elem) for elem in rec ] )
        header = ["epoch"]
        for key in self.train_keys:
            header.append("train_%s" % key)
        for key in self.train_keys:
            header.append("valid_%s" % key)
        header.append("lr")
        header.append("time")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.dd['out']), "%s/gen.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.disc), "%s/disc.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.disc2), "%s/disc2.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "a" if resume else "wb")
        if not resume:
            f.write(",".join(header)+"\n"); f.flush()
            print ",".join(header)
        #cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        if self.verbose:
            print "training..."
        for e in range(num_epochs):
            if e+1 in schedule:
                self.lr.set_value( schedule[e+1] )
            out_str = []
            out_str.append(str(e+1))
            t0 = time()
            # training
            results = _loop(self.train_fn, it_train)
            for i in range(len(results)):
                out_str.append(str(results[i]))
            #if reduce_on_plateau:
            #    cb.on_epoch_end(np.mean(recon_losses), e+1)
            # validation
            results = _loop(self.loss_fn, it_val)
            for i in range(len(results)):
                out_str.append(str(results[i]))
            out_str.append(str(self.lr.get_value()))
            out_str.append(str(time()-t0))
            out_str = ",".join(out_str)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            dump_train = "%s/dump_train" % out_dir
            dump_valid = "%s/dump_valid" % out_dir
            for path in [dump_train, dump_valid]:
                if not os.path.exists(path):
                    os.makedirs(path)
            self.plot(it_train, "%s/remove_%i.png" % (dump_train,e+1), mode='remove')
            self.plot(it_train, "%s/add_%i.png" % (dump_train,e+1), mode='add')
            self.plot(it_val, "%s/remove_%i.png" % (dump_valid,e+1), mode='remove')
            self.plot(it_val, "%s/add_%i.png" % (dump_valid,e+1), mode='add')
            if model_dir != None and (e+1) % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))

    def plot(self, itr, out_file, grid_size=10, mode='remove', deterministic=True):
        assert mode in ['remove', 'add']
        zero_fn = self.zero_fn if not deterministic else self_zero_fn_det
        enc_fn = self.enc_fn if not deterministic else self.enc_fn_det
        # remove = decompose Au into [A,u], change u -> 0 then decode
        # add = decompose B0 into [B,0], change 0 -> randomly sampled u then decode
        im_dim = self.im_shp[-1]
        grid = np.zeros((im_dim*grid_size, im_dim*grid_size, self.im_shp[0]), dtype="float32")
        for i in range(grid_size):
            for j in range(grid_size):
                this_Au, this_B0 = itr.next()
                if mode == 'remove':
                    # ok, go from Au to A0
                    target = zero_fn(this_Au)
                else:
                    # ok, go from B0 to Ba
                    # TODO: figure out bottleneck size
                    #a_fake = floatX(np.random.normal(0,1,size=(this_B0.shape[0],self.latent_dim)))
                    #a_actual = a_fake
                    # get from real latent factors from the Au data
                    _, a_actual = enc_fn(this_Au)
                    # then use those to replace the '0' factor for B0
                    target = self.dec_use_a_fn(this_B0, a_actual)
                grid[i*im_dim:(i+1)*im_dim, j*im_dim:(j+1)*im_dim, :] = target[0].swapaxes(0,1).swapaxes(1,2)
        from skimage.io import imsave
        if self.im_shp[0]==1:
            imsave(arr=grid[:,:,0],fname=out_file)
        else:
            imsave(arr=grid,fname=out_file)
                           
if __name__ == '__main__':

    def _preset(seed):
        lasagne.random.set_rng( np.random.RandomState(seed) )
        np.random.seed(seed)        

    def get_dr_iterators(batch_size):
        from iterators import BasicHdf5Iterator
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        it_train = BasicHdf5Iterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1=0, c2=4,
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        it_val = BasicHdf5Iterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1=0, c2=4,
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        return it_train, it_val
        
    def mnist(mode,seed):
        _preset(seed)
        itr_train = MnistIterator('train',9,0,32,True)
        itr_valid = MnistIterator('valid',9,0,32,True)
        # lr
        encode_decode_params = {'enc_size':64, 'feat_size':64}
        model = GeneGAN(
            generator_fn=mnist_encoder,
            generator_params=encode_decode_params,
            discriminator_fn=mnist_discriminator,
            discriminator_params={},
            opt_args={'learning_rate':theano.shared(floatX(2e-4))})
        name = "deleteme_64c_repeat"
        if mode == "train":
            model.train(it_train=itr_train, it_val=itr_valid, batch_size=128, num_epochs=1000,
                out_dir="output/%s" % name, model_dir="models/%s" % name, schedule={300: 2e-5})
    
    def dr(mode,seed):
        from architectures import g_unet_256, discriminator
        _preset(seed)
        bs = 16
        itr_train, itr_valid = get_dr_iterators(bs)
        model = GeneGAN(
            generator_fn=g_unet_256,
            generator_params={},
            discriminator_fn=discriminator,
            discriminator_params={'in_shp': (3,256,256)},
            opt_args={'learning_rate':theano.shared(floatX(2e-4))})
        name = "dr_test1"
        if mode == "train":
            model.train(it_train=itr_train, it_val=itr_valid, batch_size=bs, num_epochs=1000,
                out_dir="output/%s" % name, model_dir="models/%s" % name, schedule={300: 2e-5})
        
        
    locals()[ sys.argv[1] ](sys.argv[2], int(sys.argv[3]))

    
