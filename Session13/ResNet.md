# ResNet

Why deep, deeper , deepest ?

A Single Layer is enough  given the resource, but leads to overfitting, hence need to go deeper.

Need to go deeper ?

We cant go wide or less layer receptive field with larger kernel- but  OOM

So we go deeper with smaller kernel values and find out edges and gradients, textures and patterns, part of objects, objects - generating receptive field maps.

Convolutions are similar to correlation, and this correlation is done  w.r.t. a receptive field map. If the Receptive Field map is large, then  it would contain different templates for our object. This requires us to add more layers. 

SO basically breakdown the image into smallest piece and rebuild until they make sense. 



---



When a kernel convolves on an image it will convolve and extract what it needs only.

SO a 32*32\*3 image will be convolved by 32 kernels.

Each of these 32 kernles will conv0  





