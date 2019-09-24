# Session 1

###  Images -

TL;DR - 

​	Images are made up of overlapping similar features.
​	Similar Features combined are called a channel.

#### Breakdown 

1. Images are combination of pixels, pixels individually is just a dot/square of a certain ratio of RGB channel.

   ![Original Image](https://2.bp.blogspot.com/-tkggOOkYd00/Voln4EZhHrI/AAAAAAAAAVg/s81ocGoNFcc/s400/mandrill.png)

   ![RGB CHANNELS](https://4.bp.blogspot.com/-kAtLMhm4lCM/VolnvzNgtdI/AAAAAAAAAVY/EMESzQKQdYo/s640/mandrill_rgb.png)

2. Similar pixels together may form a patter, like part of a straight line or part of curve.

    <img src="https://upload.wikimedia.org/wikipedia/commons/2/2b/Pixel-example.png" alt="Image Breakdown"  />

   For example, looking at very zoomed in picture the edge of keyboard is a continuous collection of horizontal lines at a certain angle and the stand of monitor forms a curve line. 



#### Kernels and Channels

1. Hence an image can be broken down into set of features, and what detects these features are called **feature extractor**, **filters** or **kernels**.

2. Each kernel extract only one type of feature in an entire image and collection of similar features is called a **channel**.

   - Features of similar type for example all edges of $45^\circ$ or all edges of $90^\circ$  grouped together in one channel called a $45^\circ$ edge detector or a vertical edge detector respectively.

   <img src="https://qph.fs.quoracdn.net/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5-c" alt="Kernels"  />

   <img src="https://wiki.tum.de/download/thumbnails/23572254/filter%20levels.png?version=1&amp;modificationDate=1485348352200&amp;api=v2" style="zoom: 150%;" />

3. Combination of different channels combine to form complex features like a $\frac{1}{4} ^{th}$ of circle or  corner of a window, a cross or a plus sign.

>  Complex things are made from small features. Similarly images can be imagined as combination ( in ascending order) of   features, gradients ,textures,  patterns, parts of objects, object and scenery 



Convolution, Layers, Max Pooling 