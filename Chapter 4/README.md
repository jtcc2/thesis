# Algorithms for Finding Orientations of Supersingular Elliptic Curves and Quaternion Orders.

This is a copy of the code for the paper [Finding Orientations of Supersingular Elliptic Curves and Quaternion Orders](https://eprint.iacr.org/2023/1268) which is also available [here](https://github.com/jtcc2/finding-orientations).
See the paper for the list of authors.  

It includes:

1. Algorithms for the *Quaternion Embedding Problem* - Finding embeddings of a quadratic order within a maximal quaternion order in $B_{p,\infty}$. 
    + Includes rerandomization of the basis of the order, making it very fast for any quadratic order with discriminant less than $O(p)$.
    + Algorithm for checking if a solution gives a primitive embedding.
    + Experimental results on finding primitive embeddings (i.e. orientations) as described in the paper.
2. Examples checking properties of the Hermite normal form of basis of quaternion orders.

The code is explained with examples in jupyter notebook files. Algorithm numbering follows the numbering used in the paper.