# Session 4



## Fully Connected Layers

2D to 1D is bad. Computationally expensive and not very accurate.  For example - A vertical and  45 degree edge can give same answer by flattening the image.

Extending to next layer will give us Fully connected layer. 





VGA was the last time fully connected layers were used.

1X1 was not invented 

Only Convolution and Max pooling with Softmax and FCL was used.

## Softmax

Is Softmax BAD? 
Changes result by scaling  them so they add up to 1.

 

In real life scenarios like  Uber self driving car deciding weather the road is empty or there a pedestrian on the road. and we use softmax. What will be the result ? There was a case on Uber.





Train a model where difference is  high between prediction

## 8 codes for  NN

Global Average Pooling. is good good.





**BATCH NORMALIZTION** - Normalizing image 



 **Over Fitting** - When training accuracy is greater than validation accuracy.

An image with multiple objects - example of frog and BMW.

**DROPOUT.** - From each layer decide not to include  a few pixels or neurons while training.

What's happening is that model is not learning some part or object, or model is not learning some features. But still categorizing the image correctly.


As a result model sees incomplete objects and learn that this image is one of the label.

Its good when we have some parts that are hidden of an object. for example face of dog is hidden behind a sofa then. Dropout will help to recognize the dog even when we have not seen the complete image.

**Learning Rate** -

## Assignment

1. How many layers,
2. MaxPooling,
3. 1x1 Convolutions,
4. 3x3 Convolutions,
5. Receptive Field,
6. SoftMax,
7. Learning Rate,
8. Kernels and how do we decide the number of kernels?
9. Batch Normalization,
10. Image Normalization,
11. Position of MaxPooling,
12. Concept of Transition Layers,
13. Position of Transition Layer,
14. Number of Epochs and when to increase them,
15. DropOut
16. When do we introduce DropOut, or when do we know we have some overfitting
17. The distance of MaxPooling from Prediction,
18. The distance of Batch Normalization from Prediction,
19. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
20. How do we know our network is not going well, comparatively, very early
21. Batch Size, and effects of batch size
22. When to add validation checks
23. LR schedule and concept behind it
24. Adam vs SGD
25. etc (you can add more if we missed it here)