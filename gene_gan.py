import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
leaky_relu = LeakyRectify(0.2)
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

def encoder(enc_size=64, feat_size=64):
    l_in_conv = InputLayer((None,1,28,28))
    conv_layer = l_in_conv
    for i in range(3):
        conv_layer = batch_norm(
            Conv2DLayer(conv_layer, num_filters=(i+1)*32, filter_size=3, stride=2, nonlinearity=leaky_relu)
        )
    conv_layer = batch_norm(
        Conv2DLayer(conv_layer, num_filters=(i+1)*32, filter_size=2, nonlinearity=leaky_relu)
    )
    l_enc = DenseLayer(conv_layer, num_units=enc_size, nonlinearity=leaky_relu)
    l_feat = DenseLayer(conv_layer, num_units=feat_size, nonlinearity=leaky_relu)
    return {"l_enc": l_enc, "l_feat": l_feat}

def decoder(enc_size=64, feat_size=64):
    l_in_enc = InputLayer((None, enc_size))
    l_in_feat = InputLayer((None, feat_size))
    l_concat = ConcatLayer((l_in_enc, l_in_feat))
    l_dense = batch_norm(DenseLayer(l_concat, 96*1*1, nonlinearity=leaky_relu))
    l_reshape = ReshapeLayer(l_dense, (-1, 96, 1, 1))
    deconv_layer = l_reshape
    for i in range(3, 0, -1):
        deconv_layer = batch_norm(
            Deconv2DLayer(deconv_layer, filter_size=3, num_filters=i*32, stride=2, nonlinearity=leaky_relu)
        )
    deconv_layer = Deconv2DLayer(
        deconv_layer, filter_size=4, num_filters=1, stride=2, crop=2, nonlinearity=sigmoid)
    return {"l_in_enc": l_in_enc, "l_in_feat": l_in_feat, "out": deconv_layer}

def discriminator():
    l_in_conv = InputLayer((None,1,28,28))
    conv_layer = l_in_conv
    for i in range(3):
        conv_layer = Conv2DLayer(
            conv_layer, num_filters=(i+1)*16, filter_size=3, stride=2, nonlinearity=leaky_relu)
    conv_layer = Conv2DLayer(
        conv_layer, num_filters=(i+1)*16, filter_size=2, nonlinearity=leaky_relu)
    l_sigm = DenseLayer(conv_layer, num_units=1, nonlinearity=sigmoid)
    return l_sigm

class GeneGAN():
    def print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer.__class__.__name__, layer.output_shape, \
            "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# params:", count_params(l_out)
    def __init__(self,
                 encoder_fn, decoder_fn,
                 encoder_params, decoder_params,
                 discriminator_fn, discriminator_params,
                 lambda_recon=1.,
                 opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 reconstruction='l1', lsgan=False, verbose=True):
        self.verbose = verbose
        # get the networks for the p2p network
        dd = encoder(**encoder_params)
        dd['dummy_out'] = ConcatLayer( [dd['l_enc'], dd['l_feat']] )
        dd_dec = decoder(**decoder_params)
        if verbose:
            self.print_network(dd['dummy_out'])
            self.print_network(dd_dec['out']) # TODO: concat
        Au = T.tensor4('Au')
        B0 = T.tensor4('B0')
        a = T.fmatrix('a') # let the user input their own latent vector
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = binary_crossentropy
        # generative stuff for Au
        A_for_Au, u_for_Au = get_output([dd['l_enc'], dd['l_feat']], Au)
        # **det stuff**
        A_for_Au_det, u_for_Au_det = get_output([dd['l_enc'], dd['l_feat']], Au, deterministic=True)
        # *************
        decode_into_Au = get_output(
            dd_dec['out'], 
            {dd_dec['l_in_enc']: A_for_Au, dd_dec['l_in_feat']: u_for_Au}
        )
        decode_into_A0 = get_output(
            dd_dec['out'],
            {dd_dec['l_in_enc']: A_for_Au, dd_dec['l_in_feat']: T.zeros_like(u_for_Au)}
        )
        decode_into_Au_using_a = get_output(
            dd_dec['out'], 
            {dd_dec['l_in_enc']: A_for_Au, dd_dec['l_in_feat']: a}
        )
        Au_recon_loss = T.abs_(decode_into_Au - Au).mean()
        # discriminator stuff for Au
        disc = discriminator(**discriminator_params)
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
        B_for_B0, eps_for_B0 = get_output([dd['l_enc'], dd['l_feat']], B0)
        decode_into_Bu = get_output(
            dd_dec['out'], 
            {dd_dec['l_in_enc']: B_for_B0, dd_dec['l_in_feat']: u_for_Au}
        )
        decode_into_B0 = get_output(
            dd_dec['out'], 
            {dd_dec['l_in_enc']: B_for_B0, dd_dec['l_in_feat']: T.zeros_like(u_for_Au)}
        )
        eps_loss = T.abs_(eps_for_B0).mean()
        B0_recon_loss = T.abs_(decode_into_B0 - B0).mean()
        disc2 = discriminator(**discriminator_params)
        disc_for_Au = get_output(disc2, decode_into_Au)
        disc_for_Bu = get_output(disc2, decode_into_Bu)
        # the discriminator returns the probability the image
        # is a ground truth, i.e., it is Au, and not the predicted Bu
        disc_2_loss = adv_loss(disc_for_Au, 1.).mean() + adv_loss(disc_for_Bu, 0.).mean()
        Bu_generator_loss = adv_loss(disc_for_Bu, 1.).mean()
        # **Total loss: reconstruction error + GAN loss to distinguish Bu (fake) and Au (real)**
        total_generator_loss_2 = Bu_generator_loss + B0_recon_loss + eps_loss
        # TOTAL GENERATOR LOSS
        total_loss = total_generator_loss + total_generator_loss_2
        # -----------------------------
        gen_params = get_all_params(dd['dummy_out'], trainable=True)
        disc_params = get_all_params(dd_dec['out'], trainable=True)
        tot_gen_params = gen_params + disc_params
        disc1_params = get_all_params(disc, trainable=True)
        disc2_params = get_all_params(disc2, trainable=True)
        gen_updates = opt(total_loss, tot_gen_params, **opt_args)
        gen_updates.update(opt(disc_1_loss, disc1_params, **opt_args))
        gen_updates.update(opt(disc_2_loss, disc2_params, **opt_args))
        keys = [A0_generator_loss, Bu_generator_loss, Au_recon_loss, B0_recon_loss, eps_loss, disc_1_loss, disc_2_loss]
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
        # decompose Au into [A,0], replace 0 with a (input into the function),
        # then decode
        self.dec_use_a_fn = theano.function([Au, a], decode_into_Au_using_a)
        # ------------
        self.lr = opt_args['learning_rate']
        self.train_keys = ['Au_gen', 'Bu_gen', 'Au_recon', 'B0_recon', 'eps', 'B0_A0_disc', 'Au_Bu_disc'] # TODO:
        self.dd = dd
        self.dd_dec = dd_dec
        self.disc = disc
        self.disc2 = disc2
    def save_model(self, filename):
        with gzip.open(filename, "wb") as g:
            pickle.dump({
                'gen': get_all_param_values(self.dd['dummy_out']),
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
            set_all_param_values(self.dd['dummy_out'], dd['gen'])
            set_all_param_values(self.dd['disc'], dd['disc'])                
            set_all_param_values(self.dd['disc2'], dd['disc2'])
            
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
                draw_to_file(get_all_layers(self.atob['dummy_out']), "%s/gen.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.atob['disc']), "%s/disc.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.btoa['disc2']), "%s/disc2.png" % out_dir, verbose=True)
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
            """
            # plot nice grids
            plot_grid("%s/atob_%i.png" % (out_dir,e+1), it_val, self.atob_fn, invert=False, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            plot_grid("%s/btoa_%i.png" % (out_dir,e+1), it_val, self.btoa_fn, invert=True, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            # plot big pictures of predict(A) in the valid set
            self.generate_atobs(it_train, 1, batch_size, "%s/dump_train" % out_dir, deterministic=False)
            self.generate_atobs(it_val, 1, batch_size, "%s/dump_valid" % out_dir, deterministic=False)
            """
            if model_dir != None and (e+1) % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))
class MnistIterator():
    def _iterator(self,X_Au, X_B0, bs, shuffle):
        if shuffle:
            np.random.shuffle(X_Au)
            np.random.shuffle(X_B0)
        while True:
            for b in range(self.N // bs):
                yield X_Au[b*bs:(b+1)*bs], X_B0[b*bs:(b+1)*bs]
    def __init__(self, c1, c2, bs, shuffle):
        from load_mnist import load_dataset
        X_train, y_train, _, _, _, _ = load_dataset()
        X_Au = X_train[y_train == c1]
        X_B0 = X_train[y_train == c2]
        self.fn = self._iterator(X_Au, X_B0, bs, shuffle)
        self.N = min(X_Au.shape[0], X_B0.shape[0])
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()
            
if __name__ == '__main__':
    
    itr = MnistIterator(9,0,32,True)
    model = GeneGAN(
        encoder_fn=encoder,
        decoder_fn=decoder,
        encoder_params={'enc_size':64, 'feat_size':64},
        decoder_params={'enc_size':64, 'feat_size':64},
        discriminator_fn=discriminator,
        discriminator_params={})
    model.train(it_train=itr, it_val=itr, batch_size=32, num_epochs=100,
                out_dir="output/deleteme")
    