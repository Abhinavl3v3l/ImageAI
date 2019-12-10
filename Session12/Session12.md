# Super Convergence

Finding a good Learning Rate. 

Learning Rate can be anywhere from and can be increased from 0 - 10



Recently it was shown that using certain hyper-parameter values, using a very large learning rates with the cyclical learning rate (CLR) method, we can speed up training by an order of magnitude. This phenomenon is  called  **super convergence** .



Starting Learning Rate - How to find an optimal starting learning rate.

If set too low - Time consuming due to small weights updates.





---



How  to reach super convergence ?

By finding optimal learning rate.  An optimal learning rate will decrease the loss significantly 



A systematic approach would be  by finding relation between learning rate and loss obtained wrt that learning rate. 

​	Learning rate : Loss

if 100K images and batch size of 100 we will run it 1000 times or 1000 Iterations 





![i](C:\Users\level\Documents\GitHub\ImageAI\Session12\i.PNG)

What values if changes in theta affects the most.

For example if our optimum theta value is 0.00032 and our $\alpha$ is 0.01 and loss/gradient received is 0.01

For $\alpha = 0.01$ and loss value  $= 0.01$ Output =  0.0001 with a difference of 0.0002

For $\alpha = 0.02$ and loss value  $= 0.01$ Output =  0.0002 with a difference of 0.0001

