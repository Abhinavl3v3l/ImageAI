Session 1

###  Images -

TL;DR - 

​	Images are made up of overlapping similar features.
​	Similar Features combined are called a channel.

#### Breakdown 

1. Images are combination of pixels, pixels individually is just a dot/square of a certain ratio of RGB channel.
   	Example : There are certain values associated to each  pixels of R, G and B channels. 

   ![Original Image](https://2.bp.blogspot.com/-tkggOOkYd00/Voln4EZhHrI/AAAAAAAAAVg/s81ocGoNFcc/s400/mandrill.png)

   ![RGB CHANNELS](https://4.bp.blogspot.com/-kAtLMhm4lCM/VolnvzNgtdI/AAAAAAAAAVY/EMESzQKQdYo/s640/mandrill_rgb.png)

2. Similar pixels together may form a patter and textures, like part of a straight line or part of curve and  If you see carefully you will also notice gradients.

3. Similar pixels together may form textures, like blue screen or baboobs facial hair respectively. A dogs fur, a cats fur, sky, water, metal, skin, etc

   <img src="https://upload.wikimedia.org/wikipedia/commons/2/2b/Pixel-example.png" alt="Image Breakdown"  />

   For example, looking at very zoomed in picture the edge of keyboard is a continuous collection of horizontal lines at a certain angle and the stand of monitor forms a curve line. 

> Pixels form patters , gradients. Second level or higher level would be textures.

#### Kernels and Channels

1. Hence an image can be broken down into set of features, and what detects these features are called **feature extractor**, **filters** or **kernels**.

2. Each kernel extract only one type of feature in an entire image and collection of similar features is called a **channel**.

   - Features of similar type for example all edges of $45^\circ$ or all edges of $90^\circ$  grouped together in one channel called a $45^\circ$ edge detector or a vertical edge detector respectively.

   <img src="https://qph.fs.quoracdn.net/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5-c" alt="Kernels"  />

   <img src="https://wiki.tum.de/download/thumbnails/23572254/filter%20levels.png?version=1&amp;modificationDate=1485348352200&amp;api=v2" style="zoom: 150%;" />

3. Combination of different channels combine to form complex features like a $\frac{1}{4} ^{th}$ of circle or  corner of a window, a cross or a plus sign.

>  Complex things are made from small features. Similarly images can be imagined as combination ( in ascending order) of   features, gradients ,textures,  patterns, parts of objects, object and scenery 


TODO: Convolution, Layers, Max Pooling 

### Convolution - 

Each pixel in each channel has a value between $0-255$.
We can extract similar feature using feature extractor using convolution operation.

**Image Kernel** - An image kernel is a **$m*m$** matrix with values in  $\real$. Where $\real$ is set of all real numbers.

Example of image kernel -   

~~~python
k = [[-1, 0, 1],
	 [-1, 0, 1],
	 [-1, 0, 1]]
~~~

Example of  image - 
![img](data:image/png;base64,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)



Same image as number-   28*28 Image, Each pixel a value.
![](C:\Users\level\Documents\GitHub\ImageAI\Session2\Image5.PNG)

Convolving is a Mathematical Operation. Imagine ovelapping  $3*3$  kernel

Following kernel would convolve over image 

~~~python
# Taking an example if 3x3  matrix on which our 3x3 kernel will convolve and understanding what our Kerne does to it.
print('Understanding Convolution\n')
a = [[234, 234 ,234],[234 ,234 ,234],[223 ,242, 242]]                            
b = [[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]
print('\nConvolve of kernel',b,' in values of same range\n')

print('a = ',a)
print('b = ',b)
print('Result of element wise multiplication')
print(np.multiply(a,b))
print("Convolve Operation throws a very dark value : ",np.sum(np.multiply(a,b))) #1 Convolve Operation
print("Hence after convolvin the value is very dark since first and last columns are being negated")

print('\n\n')

## Convolving on letter H 
print('Starting of letter H from Left to Right')
a =[[232, 232 ,42],[234 ,234 ,34],[223 ,242, 42]]
print('a = ',a)
print('b = ',b)
print('Result of element wise multiplication')
print(np.multiply(a,b))
print('Starting of letter H, Kernel has seen one column worth of letter\'H\'')
print("Convolve Opeatation : ",np.sum(np.multiply(a,b)))
print("Hence after convolvin the value is  large negative value hence a very dark pixel")

print('\n\n')

# Inside Letter H (all dark values)
print('Inside Letter H')
a =[[45, 45 ,42],[45 ,60 ,40],[30 ,52, 42]]
print('a = ',a)
print('b = ',b)
print('Result of element wise multiplication')
print(np.multiply(a,b))
print("Convolve Opeatation : ",np.sum(np.multiply(a,b)))

print('\n\n')

#Exiting Letter H
print('Exiting Letter H')
a =[[45, 45 ,242],[45 ,60 ,240],[30 ,52,242]]
print('a = ',a)
print('b = ',b)
print('Result of element wise multiplication')
print(np.multiply(a,b))
print("Convolve Opeatation : ",np.sum(np.multiply(a,b)))
print('\n\n')
print('Kernel has -ve 0 +ve view')
print('So when convolving over the letter \'H\' we see 2 bright then a dark value.Resulting in  Initially. Same when kernel sees values inside letter H\n When the kernel leaves letter H, convolve operation would see small negative values and then very bright values \n Hence output shines until kernel sees all similar values.\n The reason we have white edge of letters detected on right side, we could have lit the left side by reversing the -1 and 1\'s position')
print('If this kernel was to be imagined it has dark dark bright.\n 1. All values in same range results in very dark value.\n 2. A transition from bright to dark will again result in very dark value. Since -1\'s column is dominant.\n 3. Only when we transition from dark toward brighter value, will the 1\'s column will have more weight.\n Hence the reason we have right edge detection. ')
a =[[32, 45 ,242],[34 ,60 ,234],[23 ,52, 242]]
print("Convolve Opeatation : ",np.sum(np.multiply(a,b)))
~~~

~~~
MINI OUTPUT
a =  [[45, 45, 42], [45, 60, 40], [30, 52, 42]]
b =  [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
Each element multiplied and then summed.
[[-45   0  42]
 [-45   0  40]
 [-30   0  42]]
Convolve Opeatation :  4
~~~

---



Layers
MAx Pooling

