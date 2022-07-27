# Interpretability

## Contextual Decomposition (CD)
[paper: only for LSTMs](https://arxiv.org/pdf/1801.05453.pdf)

[paper: generalized on any DNN](https://arxiv.org/pdf/1806.05337.pdf)

Gives a score to a subset $x$ of all the inputs.

We call $g_i(x)$ the output of layer $i$.
Then we define $g_i^{CD}(x) = (\Beta_i(x), \gamma_i(x))$ so $g_i(x) = \Beta_i(x) + \gamma_i(x))$ .

The final score can be generated recursively.

Implementation with pytorch: check ` cd_generic` function in [here](https://github.com/csinva/hierarchical-dnn-interpretations/blob/f3a79868420a9f51c825085d62bdff16f9e1a8f3/acd/scores/cd.py#L70)

## Agglomerative contextual decompotition

Runs hiererchical clustering on the features using CD scores as joining metric.