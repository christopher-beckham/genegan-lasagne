# genegan-lasagne
An implementation of GeneGAN in Lasagne

This is a WIP. Currently there is only a notebook with an implementation of it for MNIST. Basically, we have two sets of images --
ones with 9's, and ones with 0's. We assume that the 9 digit can be decomposed into two sets of latent variables [A,u], where
`u` is the latent variable of interest (in our case, the vertical line distinguishing a 9 from a 0), and we also assume the 0
can be decomposed into two latent variable sets [B,0] (where B may be == A, and 0 is a null set).

We can then 'encode' the 9 into [A,u], null the `u` vector, decode, and get a 0! Likewise, we can encode a 0 into [A,0], then
replace the `0` with a `u`, and it will decode into a 9. Some pics of this are shown in the notebook.

For more information, see: https://arxiv.org/pdf/1705.04932.pdf
