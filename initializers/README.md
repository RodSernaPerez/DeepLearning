# Initializers

Weights for variable must be random variables (in bias, they can be constant).
If they are too large, model could learn too slowly.


## HE Normal

Used with relu as a function activation.
The weights are obtained from:

$W \sim N(0, 1) * \sqrt{\frac{2}{n}}$

Being $n$ the number of incoming features.

## Xavier Normal

Similar to HE, but uses:

$W \sim N(0, 1) * \frac{1}{\sqrt(n}$



