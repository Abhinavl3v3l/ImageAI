# Session 2

## Why do we add layers ? Revised

1. To see the full image befor identifying what object is FOR NOW we assume object is size of image. Like CIFAR or MNIST.



---

### Receptive Field. 

---





### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)


No of Step to reach 1x1  = $99^{th}$ step

Calculation :

| Sno                                                          | Image size (x and y)                                         | Convolution                                                  | Resultant Image (x and y)                                    | Receptive Field                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1<br/>2<br/>3<br/>4<br/>5<br/>6<br/>7<br/>8<br/>9<br/>10<br/>11<br/>12<br/>13<br/>14<br/>15<br/>16<br/>17<br/>18<br/>19<br/>20<br/>21<br/>22<br/>23<br/>24<br/>25<br/>26<br/>27<br/>28<br/>29<br/>30<br/>31<br/>32<br/>33<br/>34<br/>35<br/>36<br/>37<br/>38<br/>39<br/>40<br/>41<br/>42<br/>43<br/>44<br/>45<br/>46<br/>47<br/>48<br/>49<br/>50<br/>51<br/>52<br/>53<br/>54<br/>55<br/>56<br/>57<br/>58<br/>59<br/>60<br/>61<br/>62<br/>63<br/>64<br/>65<br/>66<br/>67<br/>68<br/>69<br/>70<br/>71<br/>72<br/>73<br/>74<br/>75<br/>76<br/>77<br/>78<br/>79<br/>80<br/>81<br/>82<br/>83<br/>84<br/>85<br/>86<br/>87<br/>88<br/>89<br/>90<br/>91<br/>92<br/>93<br/>94<br/>95<br/>96<br/>97<br/>98<br/>99<br/>==100==<br/>101 | 199<br/>199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>==3==<br/>1 | 3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>==3x3== | 199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>3<br/>==1==<br/>1 | 1<br/>3<br/>5<br/>7<br/>9<br/>11<br/>13<br/>15<br/>17<br/>19<br/>21<br/>23<br/>25<br/>27<br/>29<br/>31<br/>33<br/>35<br/>37<br/>39<br/>41<br/>43<br/>45<br/>47<br/>49<br/>51<br/>53<br/>55<br/>57<br/>59<br/>61<br/>63<br/>65<br/>67<br/>69<br/>71<br/>73<br/>75<br/>77<br/>79<br/>81<br/>83<br/>85<br/>87<br/>89<br/>91<br/>93<br/>95<br/>97<br/>99<br/>101<br/>103<br/>105<br/>107<br/>109<br/>111<br/>113<br/>115<br/>117<br/>119<br/>121<br/>123<br/>125<br/>127<br/>129<br/>131<br/>133<br/>135<br/>137<br/>139<br/>141<br/>143<br/>145<br/>147<br/>149<br/>151<br/>153<br/>155<br/>157<br/>159<br/>161<br/>163<br/>165<br/>167<br/>169<br/>171<br/>173<br/>175<br/>177<br/>179<br/>181<br/>183<br/>185<br/>187<br/>189<br/>191<br/>193<br/>195<br/>197<br/>==199==<br/>199 |

---

Adding channels into the mix of convolution.

---

# Parameters Visualized

### Number of Parameters in each step.

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
4. These 288 or 9 * 32  parameters would convolve over 1 channel of image.
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

---

### Number of Images in GPU

| Step | Image          | Kernel         | Parameters                  | Number of Images |
| ---- | -------------- | -------------- | --------------------------- | ---------------- |
| 1    | (28 * 28) * 1  | (3 * 3) * 32   | (3 * 3) * 32 * 1 + 32 = 320 | 32               |
| 2    | (26 * 26) * 32 | (3 * 3) * 64   | (3 * 3) * 64 * 32 + 64      | 64               |
| 3    | (24 * 24) * 64 | (3 * 3 ) * 128 | (3 * 3) * 128 * 64 + 128    | 128              |
| 4    | (22 * 22) *128 | (3 * 3 ) * 256 | (3 * 3) * 256 * 128 + 256   | 256              |
| 5    | (20 * 20) *256 | (3 * 3 ) * 512 | (3 * 3) * 512 * 256 + 512   | 512              |

**We have 32 + 64  + 128 + 256 + 512 = 992  Images in GPU in just 5 Steps.**  

**GPU will sing OOM.**

and we have 94 more steps to go. Solution ?

## Max Pooling

Max of 2 * 2 kernel would shine maximum of 4 value and potray it and reduce the image size to half and increase the receptive field by double.



**400x400**x3   | (3x3x3)x32    | 398x398x32   RF of 3x3

CONVOLUTION BLOCK 1 BEGINS
     398x398x32  | (3x3x32)x64   | 396x396x64  RF of 5X5
     396x396x64  | (3x3x64)x128  | 394x394x128 RF of 7X7
     394x394x128 | (3x3x128)x256 | 392x392x256 RF of 9X9
     392x392x256 | (3x3x256)x**512** | 390x390x512 RF **of 11X11**

CONVOLUTION BLOCK 1 ENDS

 

TRANSITION BLOCK 1 BEGINS
     MAXPOOLING(2x2)
     195x195x512 | **(1x1x512)x32**  | 195x195x32 RF of 22x22
TRANSITION BLOCK 1 ENDS

 

CONVOLUTION BLOCK 2 BEGINS
     195x195x32   |(3x3x32)x64    | 193x193x64   RF of 24x24
     193x193x64   |(3x3x64)x128   | 191x191x128  RF of 26x26
     191x191x128  |(3x3x128)x256  | 189x189x256  RF of 28x28
     189x189x256  |(3x3x256)x512  | 187x187x512  RF of 30x30
CONVOLUTION BLOCK 2 ENDS

 

TRANSITION BLOCK 2 BEGINS
     MAXPOOLING(2x2)
     93x93x512 | **(1x1x512)x32**  | 93x93x32 RF of 60x60
TRANSITION BLOCK 2 ENDS

 

CONVOLUTION BLOCK 3 BEGINS
     93x93x32 | (3x3x32)x64 | 91x91x64    RF of 62x62

 and so on...

---