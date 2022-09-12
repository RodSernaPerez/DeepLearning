# Learning algorithms

Deep learning networks are trained using gradient descent.

Follows the equation:

<<<<<<< HEAD
$W_{i}^{k} = W_{i - 1}^{k -1} - \alpha * \frac{\partial l}{\partial W_{i}^{k-1}}$
=======
$W_{i}^{k} = W_{i - 1}^{k -1} - \alpha * \frac{\partial l}{\partial \W_{i}^{k-1}}$
>>>>>>> 3d5f8be41518fb648d93a5fcf53d817bf1a7a553

Being $l(y, y_{pred})$ the loss function.


## Batch Gradient Descent

Gradient Descent applied on a batch of data. $l$ is now the mean of the losses for each data in the batch.

## Stochastic gradient descent

Is the same that Batch Gradient Descent with a batch size of 1. 

## Momentum

The change in the weights is a function of previous gradients also:

<<<<<<< HEAD
$v_{i}^{k} = \beta *  v_{i}^{k - 1} + (1- \beta) \frac{\partial l}{\partial W_{i}^{k}}$
=======
$v_{i}^{k} = \beta *  v_{i}^{k - 1} + (1- \beta) \frac{\partial l}{\partial \W_{i}^{k}}$
>>>>>>> 3d5f8be41518fb648d93a5fcf53d817bf1a7a553

And:
$W_{i}^{k} = W_{i - 1}^{k -1} - \alpha * v_{i}^{k}$

This makes the movement more "straight forward" to the target.

Typical values of $beta$ are between 0.8 and 0.999.

It can be easily shown that:

$v_{i}^{k} = ( 1- \beta) * \sum_{n=1}^{i} \frac{\partial l}{\partial W_{i}^{n}} * \beta^{k - i}$

So the impact of a gradient is decayed with exponential speed. This is why this is called "exponential decay".

Normally we initiate the vector $v$ with zeros. This makes:

$v_{i}^{1} = ( 1- \beta) * \frac{\partial l}{\partial W_{i}^{1}}$

Which makes the warming very slow. In order to fix this, we can set:

$v_{i}^{k} = \frac{v_{i}^{k}}{1-\beta^t}$

Being $t$ the number of steps.

This bias correction is not normally needed since the number of steps is usually very large.


## RMS prop

In RMS prop, we include the second power of the derivative in the update equation, together with momentum.

$s_{i}^{k} = \beta *  s_{i}^{k - 1} + (1- \beta) \frac{\partial l}{\partial (W_{i}^{k-1}})^2$

And:
$W_{i}^{k} = W_{i - 1}^{k -1} - \alpha * \frac{\partial l}{\partial W_{i}^{k-1}} * \frac{1}{\sqrt{s_{i}^{k}}$

Intuitively, it penalises the gradients that oscillates too much.



## Adam

Adam is only using RMS prop and momentum at the same time (using correction for both of them).

$W_{i}^{k} = W_{i - 1}^{k -1} - \alpha *  \frac{v_{i}^{k}}{\sqrt{s_{i}^{k}}$

