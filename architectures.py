import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
leaky_relu = LeakyRectify(0.2)
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from layers import BilinearUpsample2DLayer

def mnist_encoder():
    # TODO: enc/feat size not used
    l_in_conv = InputLayer((None,1,28,28))
    conv_layer = l_in_conv
    nf = 64
    for i in range(3):
        conv_layer = batch_norm(
            Conv2DLayer(conv_layer, num_filters=(i+1)*nf, filter_size=3, stride=2, nonlinearity=leaky_relu)
        )
    conv_layer = batch_norm(
        Conv2DLayer(conv_layer, num_filters=(i+1)*nf, filter_size=2, nonlinearity=leaky_relu)
    )
    nf_final = conv_layer.output_shape[1]
    feat_how_many = int(nf_final*0.25)
    l_feat = SliceLayer(conv_layer, indices=slice(0,feat_how_many), axis=1)
    l_enc = SliceLayer(conv_layer, indices=slice(feat_how_many, nf_final), axis=1)
    l_concat = ConcatLayer((l_enc, l_feat), axis=1)
    deconv_layer = l_concat
    for i in range(3, 0, -1):
        deconv_layer = batch_norm(
            Deconv2DLayer(deconv_layer, filter_size=3, num_filters=i*64, stride=2, nonlinearity=leaky_relu)
        )
    deconv_layer = Deconv2DLayer(
        deconv_layer, filter_size=4, num_filters=1, stride=2, crop=2, nonlinearity=sigmoid)
    return {"l_in": l_in_conv, "l_feat": l_feat, "l_enc": l_enc, "out": deconv_layer}

def mnist_discriminator():
    l_in_conv = InputLayer((None,1,28,28))
    conv_layer = l_in_conv
    nf = 32
    for i in range(3):
        conv_layer = batch_norm(Conv2DLayer(
            conv_layer, num_filters=(i+1)*nf, filter_size=3, stride=2, nonlinearity=leaky_relu))
    conv_layer = batch_norm(Conv2DLayer(
        conv_layer, num_filters=(i+1)*nf, filter_size=2, nonlinearity=leaky_relu))
    l_sigm = DenseLayer(conv_layer, num_units=1, nonlinearity=linear)
    return l_sigm

# ----------------------

def _remove_trainable(layer):
    for key in layer.params:
        layer.params[key].remove('trainable')

def padded_conv(nf, x):
    x = Convolution(x, nf,s=1,k=3)
    x = BatchNormLayer(x)
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    return x
        
def Convolution(layer, f, k=3, s=2, border_mode='same', **kwargs):
    return Conv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), pad=border_mode, nonlinearity=linear)

def Deconvolution(layer, f, k=2, s=2, **kwargs):
    return Deconv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), nonlinearity=linear)

def UpsampleBilinear(layer, f, s=2):
    layer = BilinearUpsample2DLayer(layer, s)
    layer = Convolution(layer, f, s=1)
    return layer

def concatenate_layers(layers, **kwargs):
    return ConcatLayer(layers, axis=1)

def resblock(layer, nf, s=1, decode=False):
    left = layer
    if not decode:
        left = Convolution(left, f=nf, s=s)
    else:
        # upsample using bilinear sampling using the
        # custom stride and have the conv's s=1
        if s > 1:
            left = BilinearUpsample2DLayer(left, s)
        left = Convolution(left, f=nf, s=1)
    left = BatchNormLayer(left)
    left = NonlinearityLayer(left, leaky_rectify)
    left = Convolution(left, f=nf, s=1) # shape-preserving, always
    left = BatchNormLayer(left)
    #
    # traditionally, i padded feature maps,
    # but here, we learn a projection
    right_ds = layer
    if not decode:
        right_ds = Convolution(right_ds, k=1, f=nf, s=s)
        right_ds = BatchNormLayer(right_ds)
    else:
        # upsample using bilinear sampling,
        # then do the 1x1 convolution to match dims
        # (don't stride the 1x1 conv, we already did
        # that with the bilinear upsample)
        right_ds = BilinearUpsample2DLayer(right_ds, s)
        right_ds = Convolution(right_ds, k=1, f=nf, s=1)
        right_ds = BatchNormLayer(right_ds)
    add = ElemwiseSumLayer([left, right_ds])
    add = NonlinearityLayer(add, leaky_rectify)
    return add

def conv_bn_relu(layer, nf, s=1, num_repeats=0):
    conv = layer
    for r in range(num_repeats+1):
        if r==0:
            conv = Convolution(conv, nf, s=s)
        else:
            conv = Convolution(conv, nf, s=1)
        conv = BatchNormLayer(conv)
        conv = NonlinearityLayer(conv, nonlinearity=leaky_rectify)
    return conv

def up_conv_bn_relu(layer, nf, num_repeats=0):
    conv = layer
    for r in range(num_repeats+1):
        if r==0:
            conv = UpsampleBilinear(conv, nf)
        else:
            conv = Convolution(conv, nf, s=1)
        conv = BatchNormLayer(conv)
        conv = NonlinearityLayer(conv, nonlinearity=leaky_rectify)
    return conv


def net_256_2_resblock(nf=64, u_split=0.25, num_repeats=0, act=tanh):
    i = InputLayer((None, 3, 256, 256))
    # 1,2,4,8,8,8,8,8
    mf = [1,2,4]
    enc = resblock(i, nf*mf[0], s=2) # 128
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[0], s=1)
    #
    enc = resblock(enc, nf*mf[1], s=2) # 64
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[1], s=1)
    #
    enc = resblock(enc, nf*mf[2], s=2) # 32
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[2], s=1)
    x = enc
    # ok split this dude up
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    dec = x
    # decode
    dec = resblock(dec, nf*mf[1], s=2, decode=True) # 64
    for r in range(num_repeats):
        dec = resblock(dec, nf*mf[1], s=1, decode=True)
    #
    dec = resblock(dec, nf*mf[0], s=2, decode=True) # 128
    for r in range(num_repeats):
        dec = resblock(dec, nf*mf[0], s=1, decode=True)
    #
    dec = UpsampleBilinear(dec, 3) # 256
    dec = NonlinearityLayer(dec, act)
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": dec}

def net_256_2(nf=64, u_split=0.25, num_repeats=1, act=tanh):
    i = InputLayer((None, 3, 256, 256))
    # 1,2,4,8,8,8,8,8
    mf = [1,2,4]
    enc = conv_bn_relu(i, nf*mf[0], num_repeats=num_repeats) # 128
    enc = conv_bn_relu(enc, nf*mf[1], num_repeats=num_repeats) # 64
    enc = conv_bn_relu(enc, nf*mf[2], num_repeats=num_repeats) # 32
    x = enc
    # ok split this dude up
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    # decode
    dec = up_conv_bn_relu(x, nf*mf[1], num_repeats=num_repeats) # 64
    dec = up_conv_bn_relu(dec, nf*mf[0], num_repeats=num_repeats) # 128
    dec = UpsampleBilinear(dec, 3) # 256
    dec = NonlinearityLayer(dec, act)
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": dec}


def net_256(nf=64, u_split=0.25, act=tanh):
    i = InputLayer((None, 3, 256, 256))
    # 1,2,4,8,8,8,8,8
    ups = UpsampleBilinear
    mf = [1,2,4,8,8]
    # 128
    conv1 = Convolution(i, nf*mf[0])
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    # 64
    conv2 = Convolution(x, nf * mf[1])
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    # 32
    conv3 = Convolution(x, nf * mf[2])
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    # 16
    conv4 = Convolution(x, nf * mf[3])
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    # 16
    conv5 = Convolution(x, nf * mf[4], k=2, s=1, border_mode='valid')
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    
    # ok split this dude up
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    
    # 16
    dconv1 = Deconvolution(x, nf * mf[3], k=2, s=1)
    dconv1 = BatchNormLayer(dconv1) #2x2
    x = dconv1
    x = NonlinearityLayer(x, leaky_rectify)
    # 32
    dconv2 = ups(x, nf * mf[2])
    dconv2 = BatchNormLayer(dconv2)
    x = dconv2
    x = NonlinearityLayer(x, leaky_rectify)
    # 64
    dconv3 = ups(x, nf * mf[1])
    dconv3 = BatchNormLayer(dconv3)
    x = dconv3
    x = NonlinearityLayer(x, leaky_rectify)
    # 128
    dconv4 = ups(x, nf * mf[0])
    dconv4 = BatchNormLayer(dconv4)
    x = dconv4
    x = NonlinearityLayer(x, leaky_rectify)
    # 256
    dconv5 = ups(x, 3)
    out = NonlinearityLayer(dconv5, act)
    
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": out}

def g_unet_256(nf=64, mul_factor=[1,2,4,8,8,8,8,8], u_split=0.25, act=tanh, dropout=0., bilinear_upsample=False, disable_skip=False):
    """
    The UNet in Costa's pix2pix implementation with some added arguments.
    is_a_grayscale:
    is_b_grayscale:
    nf: multiplier for # feature maps
    dropout: add 0.5 dropout to the first 3 conv-blocks in the decoder.
      This is based on the architecture used in the original pix2pix paper.
      No idea how it fares when combined with num_repeats...
    num_repeats:
    """
    assert len(mul_factor)==8
    if bilinear_upsample:
        ups = UpsampleBilinear
    else:
        ups = Deconvolution
    i = InputLayer((None, 3, 256, 256))
    # 1,2,4,8,8,8,8,8

    mf = mul_factor
    
    # in_ch x 256 x 256
    conv1 = Convolution(i, nf*mf[0])
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    # nf x 128 x 128
    conv2 = Convolution(x, nf * mf[1])
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    # nf*2 x 64 x 64
    conv3 = Convolution(x, nf * mf[2])
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    # nf*4 x 32 x 32
    conv4 = Convolution(x, nf * mf[3])
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    # nf*8 x 16 x 16
    conv5 = Convolution(x, nf * mf[4])
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    # nf*8 x 8 x 8
    conv6 = Convolution(x, nf * mf[5])
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    # nf*8 x 4 x 4
    conv7 = Convolution(x, nf * mf[6])
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    # nf*8 x 2 x 2
    conv8 = Convolution(x, nf * mf[7], k=2, s=1, border_mode='valid')
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)

    # ok split this dude up
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)

    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))

    x = ConcatLayer([l_feat, l_enc])
    
    # nf*8 x 1 x 1
    #dconv1 = Deconvolution(x, nf * 8,
    #                       k=2, s=1)
    dconv1 = Deconvolution(x, nf * mf[6], k=2, s=1)
    dconv1 = BatchNormLayer(dconv1) #2x2
    x = concatenate_layers([dconv1, conv7])
    if disable_skip:
        x = dconv1
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    # nf*(8 + 8) x 2 x 2
    dconv2 = ups(x, nf * mf[5])
    dconv2 = BatchNormLayer(dconv2)
    x = concatenate_layers([dconv2, DropoutLayer(conv6, dropout) if dropout>0 else conv6])
    if disable_skip:
        x = dconv2
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 4 x 4
    dconv3 = ups(x, nf * mf[4])
    dconv3 = BatchNormLayer(dconv3)
    x = concatenate_layers([dconv3, DropoutLayer(conv5, dropout) if dropout>0 else conv5])
    if disable_skip:
        x = dconv3
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 8 x 8
    dconv4 = ups(x, nf * mf[3])
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, DropoutLayer(conv4, dropout) if dropout>0 else conv4])
    if disable_skip:
        x = dconv4
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 16 x 16
    dconv5 = ups(x, nf * mf[2])
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, DropoutLayer(conv3, dropout) if dropout>0 else conv3])
    if disable_skip:
        x = dconv5
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 32 x 32
    dconv6 = ups(x, nf * mf[1])
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, DropoutLayer(conv2, dropout) if dropout>0 else conv2])
    if disable_skip:
        x = dconv6
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(4 + 4) x 64 x 64
    dconv7 = ups(x, nf * mf[0])
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, DropoutLayer(conv1, dropout) if dropout>0 else conv1])
    if disable_skip:
        x = dconv7
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(2 + 2) x 128 x 128
    dconv9 = ups(x, 3)
    # out_ch x 256 x 256
    #act = 'sigmoid' if is_binary else 'tanh'
    out = NonlinearityLayer(dconv9, act)
    
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": out}

def g_unet(nf=64, dropout=False, num_repeats=0, bilinear_upsample=False):
    """
    The UNet in Costa's pix2pix implementation with some added arguments.
    in_shp:
    nf: multiplier for # feature maps
    dropout: add 0.5 dropout to the first 3 conv-blocks in the decoder.
      This is based on the architecture used in the original pix2pix paper.
      No idea how it fares when combined with num_repeats...
    num_repeats:
    """
    #assert in_shp in [512]
    i = InputLayer( (None, 3, 512, 512) )
    # in_ch x 512 x 512
    conv1 = Convolution(i, nf)
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)    
    # nf x 256 x 256
    conv2 = Convolution(x, nf * 2)
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*2 x 128 x 128
    conv3 = Convolution(x, nf * 4)
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)
    # nf*4 x 64 x 64
    conv4 = Convolution(x, nf * 8)
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 32 x 32
    conv5 = Convolution(x, nf * 8)
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 16 x 16
    conv6 = Convolution(x, nf * 8)
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 8 x 8
    conv7 = Convolution(x, nf * 8)
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 4 x 4
    conv8 = Convolution(x, nf * 8)
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 2 x 2
    conv9 = Convolution(x, nf * 8, k=2, s=1, border_mode='valid')
    conv9 = BatchNormLayer(conv9)
    x = NonlinearityLayer(conv9, nonlinearity=leaky_rectify)
    # nf*8 x 1 x 1  
    dconv1 = Deconvolution(x, nf * 8,
                           k=2, s=1)
    dconv1 = BatchNormLayer(dconv1)
    if dropout:
        dconv1 = DropoutLayer(dconv1, p=0.5)
    x = concatenate_layers([dconv1, conv8])
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    # nf*(8 + 8) x 2 x 2
    if not bilinear_upsample:
        dconv2 = Deconvolution(x, nf * 8)
    else:
        dconv2 = BilinearUpsample2DLayer(x, 2)
        dconv2 = Convolution(dconv2, nf*8, s=1)
    dconv2 = BatchNormLayer(dconv2)
    if dropout:
        dconv2 = DropoutLayer(dconv2, p=0.5)
    x = concatenate_layers([dconv2, conv7])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 4 x 4
    if not bilinear_upsample:
        dconv3 = Deconvolution(x, nf * 8)
    else:
        dconv3 = BilinearUpsample2DLayer(x, 2)
        dconv3 = Convolution(dconv3, nf*8, s=1)
    dconv3 = BatchNormLayer(dconv3)
    if dropout:
        dconv3 = DropoutLayer(dconv3, p=0.5)
    x = concatenate_layers([dconv3, conv6])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 8 x 8
    if not bilinear_upsample:
        dconv4 = Deconvolution(x, nf * 8)
    else:
        dconv4 = BilinearUpsample2DLayer(x, 2)
        dconv4 = Convolution(dconv4, nf*8, s=1)
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, conv5])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 16 x 16
    if not bilinear_upsample:
        dconv5 = Deconvolution(x, nf * 8)
    else:
        dconv5 = BilinearUpsample2DLayer(x, 2)
        dconv5 = Convolution(dconv5, nf*8, s=1)        
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, conv4])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 32 x 32
    if not bilinear_upsample:
        dconv6 = Deconvolution(x, nf * 4)
    else:
        dconv6 = BilinearUpsample2DLayer(x, 2)
        dconv6 = Convolution(dconv6, nf*4, s=1)                
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, conv3])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(4 + 4) x 64 x 64
    if not bilinear_upsample:
        dconv7 = Deconvolution(x, nf * 2)
    else:
        dconv7 = BilinearUpsample2DLayer(x, 2)
        dconv7 = Convolution(dconv7, nf*2, s=1)
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, conv2])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(2 + 2) x 128 x 128
    if not bilinear_upsample:
        dconv8 = Deconvolution(x, nf)
    else:
        dconv8 = BilinearUpsample2DLayer(x, 2)
        dconv8 = Convolution(dconv8, nf, s=1)
    dconv8 = BatchNormLayer(dconv8)
    x = concatenate_layers([dconv8, conv1])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(1 + 1) x 256 x 256
    dconv9 = Deconvolution(x, 3)
    # out_ch x 512 x 512
    out = NonlinearityLayer(dconv9, tanh)
    return out

def discriminator(in_shp, nf=32, act=sigmoid, mul_factor=[1,2,4,8], num_repeats=0, bn=False):
    shp = [None]
    shp += list(in_shp)
    assert shp[1] in [1,3]
    i = InputLayer(shp)
    x = i
    for m in mul_factor:
        for r in range(num_repeats+1):
            x = Convolution(x, nf*m, s=2 if r == 0 else 1)
            x = NonlinearityLayer(x, leaky_rectify)
            if bn:
                x = BatchNormLayer(x)
    x = Convolution(x, 1)
    out = NonlinearityLayer(x, act)
    # 1 x 16 x 16
    return out

# -----------------------------------------------------------------------

def block3(in_shp, is_a_grayscale, is_b_grayscale, u_split, multiplier=1, **kwargs):
    m = multiplier
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    conv = i
    # encoder
    conv = conv_bn_relu(conv, nf=128*m, s=2) #128
    conv = conv_bn_relu(conv, nf=256*m, s=2) #64
    conv = conv_bn_relu(conv, nf=512*m, s=2) #32
    x = conv
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    conv = x
    # decoder
    conv = up_conv_bn_relu(conv, nf=512*m) # 64
    conv = up_conv_bn_relu(conv, nf=256*m) # 128
    conv = up_conv_bn_relu(conv, nf=128*m) # 256
    conv = Conv2DLayer(conv, num_filters=1 if is_b_grayscale else 3, filter_size=3, pad='same', stride=1,
                       nonlinearity=sigmoid if is_b_grayscale else tanh)
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": conv}


def block9(in_shp, is_a_grayscale, is_b_grayscale, u_split, multiplier=1, **kwargs):
    m = multiplier
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    conv = i
    conv = batch_norm(Conv2DLayer(conv, num_filters=32*m, filter_size=7, pad='same', nonlinearity=leaky_rectify)) # c7s1
    # encoder
    conv = conv_bn_relu(conv, nf=64*m, s=2) # d64
    conv = conv_bn_relu(conv, nf=128*m, s=2) # d128
    #
    x = conv
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    #
    conv = x
    # transformer
    for r in range(4):
        conv = resblock(conv, nf=128*m, s=1) # R 128
    x = conv_bn_relu(conv, nf=128*m, s=1)
    for r in range(4):
        conv = resblock(conv, nf=128*m, s=1) # R 128
    # decoder
    conv = up_conv_bn_relu(conv, nf=64*m) # u64
    conv = up_conv_bn_relu(conv, nf=32*m) # u32
    conv = Conv2DLayer(conv, num_filters=1 if is_b_grayscale else 3, filter_size=7, pad='same',
                       nonlinearity=sigmoid if is_b_grayscale else tanh) # c7s1
    
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": conv}

# -------

def block9_debug(in_shp, is_a_grayscale, is_b_grayscale, u_split, multiplier=1, **kwargs):
    m = multiplier
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    x = i
    nf_x = x.output_shape[1]
    num_for_feat = int(u_split*nf_x)
    num_for_enc = int((1-u_split)*nf_x)
    l_feat = SliceLayer(x, axis=1, indices=slice(0, num_for_feat))
    l_enc = SliceLayer(x, axis=1, indices=slice(num_for_feat, nf_x))
    x = ConcatLayer([l_feat, l_enc])
    return {"l_in": i, "l_feat": l_feat, "l_enc": l_enc, "out": x}    

if __name__ == '__main__':

    l_res = block9(256, False, False, 0.5)
    for layer in get_all_layers(l_res["out"]):
        print layer, layer.output_shape

    ##from nolearn.lasagne.visualize import draw_to_file
    ##draw_to_file(get_all_layers(l_res), "test_resnet.png", verbose=True)
