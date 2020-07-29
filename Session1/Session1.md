# Assignment 1

### What are Channels and Kernels (according to EVA)?

1. Images are combination of pixels, pixels individually is just a dot/square of a certain ratio of RGB channel(More than 3 channels can also used). 

![Original Image](https://2.bp.blogspot.com/-tkggOOkYd00/Voln4EZhHrI/AAAAAAAAAVg/s81ocGoNFcc/s400/mandrill.png)

![RGB CHANNELS](https://4.bp.blogspot.com/-kAtLMhm4lCM/VolnvzNgtdI/AAAAAAAAAVY/EMESzQKQdYo/s640/mandrill_rgb.png)

2. Similar pixels together may form a patter, like part of a straight line or part of curve.

<img src="https://upload.wikimedia.org/wikipedia/commons/2/2b/Pixel-example.png" alt="Image Breakdown"  />

For example, looking at very zoomed in picture the edge of keyboard is a continuous collection of horizontal lines at a certain angle and the stand of monitor forms a curve line. 



#### Kernels and Channels

1. Hence an image can be broken down into set of features, and what detects these features are called **feature extractor**, **filters** or **kernels**.

2. Each *kernel* extract only one type of feature in an entire image and collection of similar features is called a **channel**.

3. Features of similar type, for example, 

   - All edges of $45^\circ$ are grouped in together in one channel
   - all vertical edge are bagged into one channel using a  vertical edge detector respectively.

   <img src="https://qph.fs.quoracdn.net/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5-c" alt="Kernels"  />

   <img src="https://wiki.tum.de/download/thumbnails/23572254/filter%20levels.png?version=1&amp;modificationDate=1485348352200&amp;api=v2" style="zoom: 150%;" />

4. Combination of different channels combine to form complex features like a $\frac{1}{4} ^{th}$ of circle or  corner of a window, a cross or a plus sign.

> Complex things are made from small features. Similarly images can be imagined as combination ( in ascending order) of features, gradients ,textures,  patterns, parts of objects, object and scenery 

---

## How are kernels initialized?

Kernels are Initialized Randomly

### What happens during the training of a DNN?

Two major steps - Forward and Backward Propagation 



Forwards Propagation -  When images are divided into batch, Convolved upon, forming kernels, patterns, gradients, parts of objects, object , scene. These are then validated against the original image if network have identified its task as its suppose to be. 



Backward Propagation - This is rectification of kernels. Kernels did a bad job and are given feedback about their values which are then rectified to perform the task better.





This process continues until a certain percentage of accuracy is achieved.



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
![](..\Session2\Image5.PNG)

Convolving is a Mathematical Operation. Imagine overlapping  $3*3$  kernel

### Why should we only (well mostly) use 3x3 Kernels?

- **Axis of Symmetry**  - Any Odd sized imaged can be achieved. Objective is to create a kernel which detects a feature.  Due of axis of symmetry any primitive feature can easily be created. Its hard to do the same with even sized kernel say 4*4. 
  - **GOOD** - 3*3, 5\*5, ... all off number kernels.
  - **BAD** - 4*4, 6\*6,.... all even numbered kernels.
- All big-shots like Nvidia, Google, Facebook uses, hence we use it.

---

### How many times do we need to perform 3x3 convolution operation to reach Receptive field of Image i.e. 1x1 from 199x199


No of Step to reach 1x1  = $99^{th}$ step

Calculation :

| Sno                                                          | Image size (x and y)                                         | Convolution                                                  | Resultant Image (x and y)                                    | Receptive Field                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1<br/>2<br/>3<br/>4<br/>5<br/>6<br/>7<br/>8<br/>9<br/>10<br/>11<br/>12<br/>13<br/>14<br/>15<br/>16<br/>17<br/>18<br/>19<br/>20<br/>21<br/>22<br/>23<br/>24<br/>25<br/>26<br/>27<br/>28<br/>29<br/>30<br/>31<br/>32<br/>33<br/>34<br/>35<br/>36<br/>37<br/>38<br/>39<br/>40<br/>41<br/>42<br/>43<br/>44<br/>45<br/>46<br/>47<br/>48<br/>49<br/>50<br/>51<br/>52<br/>53<br/>54<br/>55<br/>56<br/>57<br/>58<br/>59<br/>60<br/>61<br/>62<br/>63<br/>64<br/>65<br/>66<br/>67<br/>68<br/>69<br/>70<br/>71<br/>72<br/>73<br/>74<br/>75<br/>76<br/>77<br/>78<br/>79<br/>80<br/>81<br/>82<br/>83<br/>84<br/>85<br/>86<br/>87<br/>88<br/>89<br/>90<br/>91<br/>92<br/>93<br/>94<br/>95<br/>96<br/>97<br/>98<br/>99<br/>==100==<br/>101 | 199<br/>199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>==3==<br/>1 | 3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>==3x3== | 199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>3<br/>==1==<br/>1 | 1<br/>3<br/>5<br/>7<br/>9<br/>11<br/>13<br/>15<br/>17<br/>19<br/>21<br/>23<br/>25<br/>27<br/>29<br/>31<br/>33<br/>35<br/>37<br/>39<br/>41<br/>43<br/>45<br/>47<br/>49<br/>51<br/>53<br/>55<br/>57<br/>59<br/>61<br/>63<br/>65<br/>67<br/>69<br/>71<br/>73<br/>75<br/>77<br/>79<br/>81<br/>83<br/>85<br/>87<br/>89<br/>91<br/>93<br/>95<br/>97<br/>99<br/>101<br/>103<br/>105<br/>107<br/>109<br/>111<br/>113<br/>115<br/>117<br/>119<br/>121<br/>123<br/>125<br/>127<br/>129<br/>131<br/>133<br/>135<br/>137<br/>139<br/>141<br/>143<br/>145<br/>147<br/>149<br/>151<br/>153<br/>155<br/>157<br/>159<br/>161<br/>163<br/>165<br/>167<br/>169<br/>171<br/>173<br/>175<br/>177<br/>179<br/>181<br/>183<br/>185<br/>187<br/>189<br/>191<br/>193<br/>195<br/>197<br/>==199==<br/>199 |

---

# No of Parameters in each Step

| Step | Image          | Kernel         | Parameters                  |
| ---- | -------------- | -------------- | --------------------------- |
| 1    | (28 * 28) * 1  | (3 * 3) * 32   | (3 * 3) * 32 * 1 + 32 = 320 |
| 2    | (26 * 26) * 32 | (3 * 3) * 64   | (3 * 3) * 64 * 32 + 64      |
| 3    | (24 * 24) * 64 | (3 * 3 ) * 128 | (3 * 3) * 128 * 64 + 128    |

---

### Step 1

| Step | Image         | Kernel       | Parameters                  |
| ---- | ------------- | ------------ | --------------------------- |
| 1    | (28 * 28) * 1 | (3 * 3) * 32 | (3 * 3) * 32 * 1 + 32 = 320 |

1. 3 * 3 Kernel would have 9 parameters. 
2. Image Kernels values for ONE channel is ONE. 
3. There are 32 such kernels in Step 1. Hence 9 * 32 parameters.
4. These 288 or 9 * 32  parameters would convolve over 1 channel of image.(GrayScale)
5. Adding 32 biases to this equation give **320** parameters to train.

> Hence m * m * c * ic is the number of parameters at a particular step. Where m is size of kernel. c is number of kernel channels and ic are number of image kernels.

---

### Step 2

| Step | Image          | Kernel       | Parameters                  |
| ---- | -------------- | ------------ | --------------------------- |
| 1    | (28 * 28) * 1  | (3 * 3) * 32 | (3 * 3) * 32 * 1 + 32 = 320 |
| 2    | (26 * 26) * 32 | (3 * 3) * 64 | (3 * 3) * 64 * 32 + 64      |

1. 3 * 3 Kernel would have 9 parameters. 
2. Image Kernels values for ONE channel is ONE. 
3. This kernel would convolve on all 32 channels.
4. Hence a kernel would convolve 32 times.
5. And there are 64 such kernels and 64 biases.
6. Hence,  **(3 * 3) * 64 * 32 + 64** parameters to train. 

