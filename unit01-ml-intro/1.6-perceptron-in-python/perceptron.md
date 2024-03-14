1.1 The perceptron cost function

With two-class classification we have a training set of $P$ points $\left\{\left(\mathbf{x}_p, \boldsymbol{y}_p\right)\right\}_{p=1}^P$ - where $y_p$ 's take on just two label values from $\{-1,+1\}$ - consisting of two classes which we would like to learn how to distinguish between automatically. As we saw in our discussion of logistic regression, in the simplest instance our two classes of data are largely separated by a hyperplane referred to as the decision boundary with each class (largely) lying on either side. Logistic regression determines such a decision boundary by fitting a nonlinear logistic regressor to the dataset, with the separating hyperplane falling out naturally where this logistic function pierces the $y=0$ plane. Specifically, we saw how the decision boundary was formally given as a hyperplane
$$
w_0+\mathbf{x}^T \mathbf{w}=0
$$

In separating the two classes we saw how this implied - when hyperplane parameters $w_0$ and $\mathbf{w}$ were chosen well - that for most points in the dataset, those from class +1 lie on the positive side of the hyperplane while those with label -1 lie on the negative side of it, i.e.,
$$
\begin{array}{ll}
w_0+\mathbf{x}_p^T \mathbf{w}>0 & \text { if } y_p=+1 \\
w_0+\mathbf{x}_p^T \mathbf{w}<0 & \text { if } y_p=-1
\end{array}
$$

While the perceptron approach works within the same framework, its approach to determining the decision boundary is more direct. With the perceptron we aim to directly determine the decision boundary by building a cost function based on these ideal properties, and whose minimum provides optimal weights that reflect these properties as best as possible.

Combining the ideal conditions above - much as we did when deriving how to predict the label value of a point with a trained logistic regressor - we can consolidate the ideal decision boundary conditions describing both classes below in a single equation as
$$
-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)<0
$$

Notice we can do so specifically because we chose the label values $y_p \in\{-1,+1\}$. Likewise by taking the maximum of this quantity and zero we can then write this condition, which states that a hyperplane correctly classifies the point $\mathbf{x}_p$, equivalently as
$$
\max \left(0,-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)\right)=0
$$

Note that the expression $\max \left(0,-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)\right)$ is always nonnegative, since it returns zero if $\mathbf{x}_p$ is classified correctly, and returns a positive value if the point is classified incorrectly. It is also a rectified linear unit - first discussed in our series on the basics of mathematical functions - being the maximum of one quantity and zero.

This expression is useful not only because it characterizes the sort of linear decision boundary we wish to have, but more importantly by simply summing it over all the points we have the nonnegative cost function
$$
g\left(w_0, \mathbf{w}\right)=\sum_{p=1}^P \max \left(0,-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)\right)
$$
whose minimum provides the ideal scenario for as many points as possible.

This cost function goes by many names such as the perceptron cost, the rectified linear unit cost (or ReLU cost for short), and the hinge cost (since when plotted a ReLU function looks like a hinge). This cost function is always convex but has only a single (discontinuous) derivative in each input dimension. This implies that we can only use gradient descent to minimize it as Newton's method requires a function to have a second derivative as well. Note that the ReLU cost also always has a trivial solution at $w_0=0$ and $\mathbf{w}=\mathbf{0}$, since indeed $g(0, \mathbf{0})=0$, thus one may need to take care in practice to avoid finding it (or a point too close to it) accidentally.

This ReLU cost function is always convex but has only a single (discontinuous) derivative in each input dimension, implying that we can only use gradient descent to minimize it. The ReLU cost always has a trivial solution at $w_0=0$ and $\mathbf{w}=\mathbf{0}$, since indeed $g(0, \mathbf{0})=0$, thus one may need to take care in practice to avoid finding it (or a point too close to it) accidentally.

Example 1: Using gradient descent to minimize the ReLU cost

In this example we use (unnormalized) gradient descent to minimize the ReLU perceptron cost function. Note however that in examining a partial derivative of just one summand of the cost with respect to weights in $\mathbf{w}$ we have
$$
\frac{\partial}{\partial w_n} \max \left(0,-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)\right)=\left\{\begin{array}{l}
-y_p x_{p, n} \text { if }-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)>0 \\
0 \quad \text { else }
\end{array}\right.
$$

Similarly for $w_0$ we can write
$$
\frac{\partial}{\partial w_0} \max \left(0,-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)\right)= \begin{cases}-y_p & \text { if }-y_p\left(w_0+\mathbf{x}_p^T \mathbf{w}\right)>0 \\ 0 & \text { else }\end{cases}
$$

We can then conclude the magnitude of the full cost function's gradient will not necessarily diminish to zero close to global minima and could stay fixed (in magnitude) based on the dataset. Thus, it is possible for gradient descent with a fixed steplength value $\alpha$ to oscillate and 'zig-zag' around, never going to a minimum (as detailed in our series on mathematical optimization). In this case we need to either tune a fixed steplength or choose a diminishing one.

### References
1. https://rezaborhani.github.io/mlr/blog_posts/Linear_Supervised_Learning/Part_3_Perceptron.html