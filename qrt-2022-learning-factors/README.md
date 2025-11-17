# Learning factors for stock market returns predictions, by QRT (2022)

https://challengedata.ens.fr/challenges/72

## Summary

The aim of this challenge is to learn one matrix $A \in \mathbb{R}^{D \times F}$
with orthonormal columns and
one vector $\beta \in \mathbb{R}^F$ which maximises the metric on predictions of
the form
$R_{t+k+1} = (R_{t+k} \dots R_{k}) \cdot A \cdot \beta$.
Here, $D = 250$ is the number of days in one input example,
$F = 10$ is the number of factors, and $R_t \in \mathbb{R}^N$ 
is the column vector of returns at time $t$ (containing the returns of $N = 50$ stocks).

Because this is a linear combination of linear factors, it suffices to learn 
a single weight vector $w = A \cdot \beta$ and fill the remaining columns of $A$ afterwards.

## Observations

### Challenge goals

It seems that the aim of the challenge was to use optimisation methods on Riemannian 
manifolds (specifically the Stiefel manifold) to simultaneously learn the $F$ 
orthonormal factors.

This can be done by building a model with $A$ and $\beta$ as in the challenge statement
and optimising the weights while enforcing the orthonormality condition in 
one of two ways (or possibly more).

#### using torch parametrisations

The torch parametrisations (see [PyTorch - Parametrizations Tutorial](https://docs.pytorch.org/tutorials/intermediate/parametrizations.html)) wrap model weight to apply certain
transformations when the weights are called upon.
Using the `orthogonal` parametrisation, we can ensure that the columns of $A$ are
orthonormal after every optimiser step, thus solving the challenge without much more
effort.

#### using optimisation on Riemannian manifolds

We can also take the optimisation steps on a given Riemannian manifold 
(here the Stiefel manifold) instead of in Euclidean space.
This is achieved by first projecting the Euclidean gradient on the tangent space, 
then taking an optimiser step, and finally retracting the result onto the manifold.

See for instance the [book by N. Boumal](https://www.cambridge.org/us/universitypress/subjects/mathematics/optimization-or-and-risk-analysis/introduction-optimization-smooth-manifolds?format=PB&isbn=9781009166157) for the general methodology and details about the Stiefel 
manifold.
The most commonly used retraction is the Cayley retraction, which can be implemented
in a computationally efficient manner using the [Sherman-Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) (or Sherman-Morrison-Woodbury formula).

Of course, neither of these methods should produce better results as the simple linear 
regression and both are more computationally expensive.

### Challenge ranking

The weights learned on the training set do not translate well to the test set 
of the challenge, which is why the competition scores are all below 1% accuracy.
Most submissions are therefore very close (in terms of score) and very slight 
changes in the submitted weights can result in quite large rank jumps on the leaderboard,
making the challenge metric quite unstable.
