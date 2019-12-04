# Session5  - Regularizations

Quiz Breakdown

Fully connected layers do not retain 

1. Spatial Information  -   2D to 1D looses all spatial information
2. Translational Invariance  -  a one and a 45 degree edge is same for dense layers
3. Rotational  Invariance -  a one and a 45 degree edge is same for dense layers
4. Size Invariance - 
5. Illumination Invariance.

![](https://i.stack.imgur.com/iY5n5.png)

## Batch Normalization and Regularization





- Redistributing data
- Normalization  vs Regularization
- How we normalize
- What network reacts to image with normalized vs un-normalized data.
- Normalizes data or values so the bowl would be circular and not extended in certain axis.
- Understanding CS - **Batch Normalization** solves a problem called **Internal Covariate shift** 
  - BN affects relu is good way
- Regularization 





### Batch Normalization -  

First Image's spectrogram, Limited area in pixel scale where pixels are located.

The is very less variance in colors which is bad for feature learning.

High variance is good.

Second image is Normalized with colors spread between, high variance .

![](http://www.giassa.net/wp-content/uploads/2010/01/hist-compare1.png)







### Image Equalization 

 ![Image result for image normalization](http://yeephycho.github.io/blog_img/normalization.jpg) 

Image Equalization is good for eye but bad for CNN.

Visualize Normalization.

![Image result for normalized image](https://s3.amazonaws.com/codecademy-content/courses/normalization/outlier.png)![Image result for normalized image](https://kharshit.github.io/img/normalization.png)

1. Calculate mean
2.  Sub mean from whole data set
3. Divide Image by standard deviation

---



#### How to Normalize data -

Mean of image - Sub all pixel and divide by number   of pixel 

1 image  3 channel - 3 mean, 3 stand deviation 

 Calculate mean of all images and divide by number of images.



---

32 batch size - 9th layer have 10 channels.

TOTAL CHANNELS =  10 * 32

32 Channel 1

32 Channel  2

32 Channel 3



and each channel has different feature.

Normalization formula.
$$
\hat{x} = \frac{x-\mu}{\sigma} \label 1
$$
$x-\mu$ is SHIFT $\sigma $ is SCALE



Initially Training parameters was using
$$
\hat y = xp \label 2
$$
where p is trainable parameter and x is input and $\hat y $ is output 

which we will further negate with actual output $y$ and then calculate loss using BP

Similarly before feeding  input $x$  to hypothesis function we normalize it using equation  $\ref 1$ hence the output of normalized image becomes 
$$
\hat y = (\frac{x-\mu}{\sigma})p \label 3
$$

This decay is applied to loss.



## L2 Regularization 

Loss  = ($y-\hat y$)

**L2 Regularization** 

Loss  = (Truth  - (Hypothesis Function)) - L2 Regularization 

Loss  = ($y$- $\hat y$) - $(\frac{\lambda}{2*m})||w^2||$

---

