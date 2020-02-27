---
layout: post
title:  "Generative methods for classification"
date:   2020-02-15 11:45:48 +0100
categories: ml
mathjax: true
---


#### Prerequisites

+ Basic calculus (logarithm rules, computing derivatives, properties of derivatives, chain rule for derivatives)
+ Linear algebra (properties of traces, matrix inverses, determinants, properties of symmetric matrices)
+ Basic probability and bayesian statistics notions (prior, posterior, Bayes' rule, joint probability, independence, probability chain rule, Gaussian distribution, categorical distribution, most of them can be found [here](https://www.cs.princeton.edu/~bee/courses/scribe/lec_08_26_2013.pdf))
+ Optimization concepts (convex and concave functions, finding extrema points, method of Lagrange multipliers)
+ Matrix calculus (gradients and their properties, some identities involving gradients)

### Probabilistic framework for generative classification

Classification is the problem of assigning a discrete label (class) to a data point. Unlike discriminant methods, where the class is assigned directly, without any measure of uncertainty, a generative approach models the joint probability of the data: 


\begin{equation}
\overbrace{p(\boldsymbol{x}, y)}^{\text{joint}} = 
\underbrace{p(y)}_{\text{class prior}} \overbrace{p(\boldsymbol{x}\mid y)}^{\text{class conditional}} \tag{1}
\end{equation}


where $$ \boldsymbol{x} \in \mathbb{R}^D $$ (note the usage of bold symbols for vectors) is a point to be classified and $$ y \in \{1, 2, \dots K \} $$ is its label, with $$ K $$ being the total number of classes.

Intuitively, the prior represents how often a class is expected to appear and the class conditional is the probability density function (assuming $$ X $$ are continous) of the data having the class indicated by y.

Using this framework, the posterior probability of each class $$ k $$ for a new point $$ \boldsymbol{x}_{new} $$ is given by Bayes' rule:

$$ 
\begin{equation}
p(y = k \mid \boldsymbol{x}_{new}) = 
\frac{p(\boldsymbol{x}_{new} \mid y = k) p(y = k)}{p(\boldsymbol{x}_{new})} \propto 
p(\boldsymbol{x}_{new} \mid y = k) p(y = k) \tag{2}
\end{equation}
$$

where we used the fact that $$ p(\boldsymbol{x}_{new}) $$ is just a scaling factor that appears in all posterior probabilities and thus can be ignored.

Now, $$ \boldsymbol{x}_{new} $$ will be assigned to the class which yields the highest posterior probability:

$$ 
\begin{equation}
y_{new} = \arg\max_{k} p(y = k \mid \boldsymbol{x}_{new}) \tag{3}
\end{equation}
$$


### Normally distributed inputs

Up to this point, we have only established the probabilistic framework we will use, but we haven't made any assumption about the distribution of the data within each class. We will continue by making some assumptions about the underlying distributions of the data and then estimating the parameters of these distributions.

#### Modelling the class priors and class conditionals

We will assume that the class conditionals are [multivariate Gaussian distributions](https://en.wikipedia.org/wiki/Multivariate_normal_distribution):

$$ 
\begin{equation}
p(\boldsymbol{x} \mid y = k) = 
\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = 
\frac{1}{(2 \pi)^{\frac{D}{2}}} \frac{1}{\det({\boldsymbol{\Sigma}_k})^{\frac{1}{2}}} e^{-\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_k)} \tag{4}
\end{equation}
$$

Because there are $$ K $$ classes and their probabilities sum to one, the natural choice for the class priors is a [categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) with parameters $$ \boldsymbol{\pi} = \begin{bmatrix} \pi_1, \pi_2, \dots, \pi_K \end{bmatrix} $$:

$$ 
\begin{equation}
p(y = k) = \pi_k, \forall k \in \{1, 2, \dots K\} \tag{5}
\end{equation}
$$

### Maximum Likelihood Estimation (MLE) of the parameters

Before moving on, let's recap what we have until now: we modeled the class priors using a categorical distribution with parameters $$ \boldsymbol{\pi} $$ and the class conditionals using Gaussian distributions with means $$ \boldsymbol{\mu}_k $$ and covariance matrices $$ \boldsymbol{\Sigma}_k, \forall k \in \{1, 2, \dots k \} $$. However, the parameters of these distributions are not known, so we must use the dataset in order to estimate them.


#### Maximum Likelihood Estimation (MLE) refresher 

MLE is a method for estimating the parameters $$ \boldsymbol{\theta} $$ of a probability distribution $$ p(\boldsymbol{x} \mid \boldsymbol{\theta}) $$ by maximizing the likelihood function of a dataset $$ \mathcal{D} = \{\boldsymbol{x}_1, \dots, \boldsymbol{x}_n\} $$ consisting of $$ n $$ i.i.d samples:


$$ 
\begin{equation}
\boldsymbol{\hat{\theta}} = 
\arg\max_{\boldsymbol{\theta}} \underbrace{\mathcal{L}( \mathcal{D} \mid \boldsymbol{\theta})}_{\text{likelihood}} =
\arg\max_{\boldsymbol{\theta}} \underbrace{\prod_{i=1}^N p(\boldsymbol{x}_i \mid \boldsymbol{\theta})}_{\text{likelihood}} \tag{6}
\end{equation}
$$

Because it is inconvenient to maximize a function containing a product, we can make use of the fact that the logarithm function transforms products into sums and is non-decreasing, thus preserving the extrema of a function:

$$ 
\begin{equation}
\boldsymbol{\hat{\theta}} =
\arg\max_{\boldsymbol{\theta}} \underbrace{\log \big( \prod_{i=1}^N p(\boldsymbol{x}_i \mid \boldsymbol{\theta}) \big)}_{\text{log likelihood}} = 
\arg\max_{\boldsymbol{\theta}} \underbrace{\sum_{i=1}^N \log p(\boldsymbol{x}_i \mid \boldsymbol{\theta})}_{\text{log likelihood}}  \tag{7}
\end{equation}
$$

Usually, when $$ \boldsymbol{\theta} $$ is unconstrained, the maximizer of the log likelihood function is found by setting the derivative w.r.t $$ \boldsymbol{\theta} $$ (or gradient if working with a multidimensional parameter) to zero and proving that the maximized function is concave, which is sufficient to conclude that extrema point found is indeed a maximizer. If $$ \boldsymbol{\theta} $$ is constrained, then a different method such as the method of Lagrange multipliers should be used.

#### Modeling the joint distribution of a sample

Using the class priors and class conditionals assumed above, the joint distribution of an input and its corresponding label can be written as:

$$ 
p(\boldsymbol{x}, y = k) = p(y = k) p(\boldsymbol{x} \mid y = k) = 
\pi_k \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = 
$$

$$
\begin{equation}
\prod_{i=1}^K (\pi_i \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i))^{I[y = i]} \tag{8}
\end{equation}
$$

where $$ I[\text{condition}] $$ is the indicator function (see [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket)) which evaluates to 1 only if the condition inside the brackets is true.

#### Log likelihood of the dataset

Assuming we have a dataset consisting of N [independent and identically distributed](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) samples, $$ \mathcal{D} = \{(\boldsymbol{x}_i, y_i)_{i=1}^N\}$$, we can write the likelihood of the data as the product of the probabilities of all samples:

$$ 
\begin{equation}
\mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
\prod_{i=1}^N \prod_{k=1}^K (\pi_k \mathcal{N}(\boldsymbol{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k))^{I[y_i = k]} \tag{9}
\end{equation}
$$

Thus, the log likelihood of the dataset will be:

$$ 
\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
\sum_{i=1}^N \sum_{k=1}^K \log(\pi_k \mathcal{N}(\boldsymbol{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k))^{I[y_i = k]} =
$$

$$	
\begin{equation}
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \big( \log\pi_k + \log\mathcal{N}(\boldsymbol{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)) \big) \tag{10}
\end{equation}
$$

#### Estimating $$ \boldsymbol{\pi}_k $$

We first notice that $$ \pi_k $$ appears in the log likelihood only through the term $$ \log \pi_k $$, so we can ignore the other term which doesn't depend on it:

$$ 
\begin{equation}
\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) \propto 
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \log\pi_k \tag{11}
\end{equation}
$$

Because the $$ \pi_k $$'s are constrained such that $$ \sum_{k=1}^K \pi_k = 1 $$ (as they are the parameters of a categorical distribution), we can use the method of Lagrange multipliers to find the maximizer of the log likelihood subject to this constraint.

We formulate the Lagrangian:

$$
\begin{equation}
\mathcal{L}(\boldsymbol{\pi}, \lambda) = 
\sum_{i=1}^N \sum_{k=1}^K \Big( I[y_i = k] \log\pi_k \Big) - \lambda (\sum_{k=1}^K \pi_k - 1) \tag{12}
\end{equation}
$$

which we can reorder a bit and unclutter by noticing that the interior sum $$ \sum_{i=1}^N  I[y_i = k] $$ is just the number of data points which have class $$ k $$, which we can denote as $$ N_k $$:

$$ 
\mathcal{L}(\boldsymbol{\pi}, \lambda) = 
\sum_{k=1}^K \log\pi_k \underbrace{\sum_{i=1}^N  I[y_i = k]}_{N_k} - \lambda (\sum_{k=1}^K \pi_k - 1) = 
$$

$$
\begin{equation}
\sum_{k=1}^K N_k \log\pi_k  - \lambda (\sum_{k=1}^K \pi_k - 1) \tag{13}
\end{equation}
$$

Taking the derivative of the Lagrangian w.r.t $$ \pi_j$$ and setting it to zero, we obtain:

$$ 
\begin{equation}
\frac{\partial{\mathcal{L}(\boldsymbol{\pi}, \lambda)}}{\partial{\pi_j}} = 
\frac{N_j}{\pi_j} - \lambda \overset{!}{=} 0 \implies \pi_j = \frac{N_j}{\lambda} \tag{14}
\end{equation}
$$

Replacing $$ \pi_j $$ back into the Lagrangian we obtain the dual:

$$ 
g(\lambda) = 
\sum_{k=1}^K N_k log\frac{N_k}{\lambda} - \lambda (\sum_{k=1}^K \frac{N_k}{\lambda} - 1) =
\sum_{k=1}^K N_k log\frac{N_k}{\lambda} + \lambda - \sum_{k=1}^K N_k \implies
$$

$$
\frac{d g(\lambda)}{d \lambda} = 
\sum_{k=1}^K N_k \frac{\lambda}{N_k} \Big[ \frac{d}{d \lambda} \frac{N_k}{\lambda} \Big] + 1 = 
\sum_{k=1}^K \frac{-N_k}{\lambda} + 1 \overset{!}{=} 0 \implies
$$

$$
\begin{equation}
\frac{1}{\lambda} \sum_{k=1}^K N_k = 1 \tag{15}
\end{equation} 
$$

We remember that $$ N_k $$ was the number of elements in the dataset belonging to class $$ k $$, so naturally the sum of all $$ N_k $$'s will be $$ N $$, the total number of elements in the dataset. Thus, we can conclude that:

$$ 
\begin{equation} 
\lambda = N \implies \pi_j = \frac{N_j}{N}  \tag{16}
\end{equation}
$$ 

So we arrived at a simple result stating that the estimated probability for each class must be the empirical probability of that class, measured on dataset $$ \mathcal{D} $$.

#### Estimating $$ \boldsymbol{\mu}_k $$

Now, we look at the log likelihood again [10] and only keep the terms containing $$ \boldsymbol{\mu}_k $$:

$$
\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) \propto
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \log\mathcal{N}(\boldsymbol{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)) = 
$$

$$
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Big( 
		\underbrace{\log \frac{1}{(2 \pi)^{\frac{D}{2}}} + 
					\log \frac{1}{\det(\boldsymbol{\Sigma_k})^{\frac{1}{2}}}}_{\text{ct. w.r.t } \boldsymbol{\mu_k}} -
		\frac{1}{2} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)
		\Big) \propto
$$

$$
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Big( 
-\frac{1}{2} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)
		\Big) \implies
$$

$$
\nabla_{\boldsymbol{\mu_j}}{\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})} =
\sum_{i=1}^N I[y_i = j]  \Big( -\frac{1}{2} 2 \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_j)
\Big) \overset{!}{=} 0 \implies
$$

$$
\sum_{i=1}^N I[y_i = j]  (\boldsymbol{x_i} - \boldsymbol{\mu}_j) = 0 \implies
\sum_{i=1}^N I[y_i = j] \boldsymbol{x_i} = \boldsymbol{\mu_j} \underbrace{\sum_{i=1}^N I[y_i = j]}_{N_j} \implies 
$$

$$
\begin{equation}
\boldsymbol{\mu_j} = \frac{1}{N_j} \sum_{i=1}^N I[y_i = j] \boldsymbol{x_i}
\tag{17}
\end{equation}
$$

Again, we arrived at a very simple result: the mean of each class conditional is the empirical mean of all data points belonging to that class.

#### Estimating $$ \boldsymbol{\Sigma}_k $$

Similarly, we find the optimal $$ \boldsymbol{\Sigma_k} $$ by computing the gradient of the log likelihood w.r.t $$ \boldsymbol{\Sigma_k} $$ and setting it to zero.

$$
\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) \propto
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \log\mathcal{N}(\boldsymbol{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)) = 
$$

$$
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Big( 
		\underbrace{\log \frac{1}{(2 \pi)^{\frac{D}{2}}}}_{\text{ct. w.r.t } \boldsymbol{\Sigma_k}} + 
					\log \frac{1}{\det(\boldsymbol{\Sigma_k})^{\frac{1}{2}}} -
		\frac{1}{2} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)
		\Big) \propto
$$

$$
\sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Big( 
					\log \frac{1}{\det(\boldsymbol{\Sigma_k})^{\frac{1}{2}}} -
		\frac{1}{2} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)
		\Big) =
$$

$$ 
\begin{equation}
\frac{1}{2} \sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Big( 
	\log \det(\boldsymbol{\Sigma_k^{-1}}) -
		(\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k)
		\Big)
\tag{18}
\end{equation}
$$

Because the function is concave in $$ \boldsymbol{\Sigma_j^{-1}} $$, we can compute the gradient w.r.t $$ \boldsymbol{\Sigma_j^{-1}} $$ directly:

$$
\nabla_{\boldsymbol{\Sigma_j^{-1}}}{\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})} = 
$$

$$
\begin{equation}
\small
\frac{1}{2} \sum_{i=1}^N \sum_{k=1}^K I[y_i = k] \Bigg( 
	\nabla_{\boldsymbol{\Sigma_j^{-1}}} \Big( \log \det(\boldsymbol{\Sigma_k^{-1}}) \Big) -
		\nabla_{\boldsymbol{\Sigma_j^{-1}}} \Big( (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k) \Big)
		\Bigg)
\tag{19}
\end{equation}
$$

We will compute the two gradients separately. For the first one we will use the log-determinant gradient rule from the table [here](https://en.wikipedia.org/wiki/Matrix_calculus?fbclid=IwAR1Pci6kG1MD87fkUfx8OZmh-YJVHLzcWBoI1dl3y5sFQb-ucvdxyWa3atQ#Scalar-by-matrix_identities), coupled with the fact that $$ \boldsymbol{\Sigma_k} $$ is a covariance matrix, thus symmetric:

$$ 
\nabla_{\boldsymbol{\Sigma_j^{-1}}} \Big( \log \det(\boldsymbol{\Sigma_k^{-1}}) \Big) 
\overset{\frac{\partial \log (\det (\boldsymbol{X}))}{\partial \boldsymbol{X}} = (\boldsymbol{X}^{-1})^T }{=\mathrel{\mkern-3mu}=\mathrel{\mkern-3mu}=}
\left\{
	\begin{array}{ll}
     0 \text{, if } j \neq k \\
     ((\boldsymbol{\Sigma_k}^{-1})^{-1})^T \text{, if } j = k
\end{array} 
\right. 
$$

$$
\begin{equation}
\overset{\boldsymbol{\Sigma_k} \text{ symm.}}{=}
\left\{
	\begin{array}{ll}
     0 \text{, if } j \neq k \\
     \boldsymbol{\Sigma_k} \text{, if } j = k
\end{array} 
\right. 
\tag{20}
\end{equation}
$$

For the second one, in a similar way, if $$ k \neq j $$, then the gradient is 0. For $$ k = j$$, we can notice that the expression under the gradient is a scalar, so we can apply the trace operator on it, as $$ Tr(a) = a, \forall a \in \mathbb{R} $$. The trace operator has some nice properties, such as the [cyclic property](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Cyclic_property) and [allowing transposing](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Basic_properties), so using the rule for the gradient of a product of two matrices w.r.t one of the matrices (from the table [here](https://en.wikipedia.org/wiki/Matrix_calculus?fbclid=IwAR1Pci6kG1MD87fkUfx8OZmh-YJVHLzcWBoI1dl3y5sFQb-ucvdxyWa3atQ#Scalar-by-matrix_identities)), we get:

$$
\nabla_{\boldsymbol{\Sigma_k^{-1}}} \Big( (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k) \Big) = 
\nabla_{\boldsymbol{\Sigma_k^{-1}}}  Tr \Big((\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k) \Big) \overset{\text{cyclic prop.}}{=}
$$

$$
\nabla_{\boldsymbol{\Sigma_k^{-1}}}  Tr \Big(\boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x_i} - \boldsymbol{\mu}_k) (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \Big) 
\overset{\frac{\partial Tr(\boldsymbol{XA})}{\partial \boldsymbol{X}} = \boldsymbol{A}^T}{=\mathrel{\mkern-3mu}=\mathrel{\mkern-3mu}=}
\Big( (\boldsymbol{x_i} - \boldsymbol{\mu}_k) (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T \Big)^T =
$$

$$
\begin{equation}
(\boldsymbol{x_i} - \boldsymbol{\mu}_k) (\boldsymbol{x_i} - \boldsymbol{\mu}_k)^T 
\tag{21}
\end{equation}
$$

Now, using the results (20) and (21) and plugging them into the expression of the gradient of the log likelihood w.r.t $$ \boldsymbol{\Sigma_j}^{-1} $$ (19), we finally get:

$$
\nabla_{\boldsymbol{\Sigma_j^{-1}}}{\log \mathcal{L}(\mathcal{D} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})} = 
\frac{1}{2} \sum_{i=1}^N I[y_i = j] \Bigg( 
	\underbrace{\boldsymbol{\Sigma_j}}_{\text{ct. w.r.t i}} -
	(\boldsymbol{x_i} - \boldsymbol{\mu}_j) (\boldsymbol{x_i} - \boldsymbol{\mu}_j)^T 	
		\Bigg)
\overset{!}{=} 0 \implies
$$

$$
\boldsymbol{\Sigma_j} \underbrace{\sum_{i=1}^N I[y_i = j]}_{N_j}  =
\sum_{i=1}^N I[y_i = j] (\boldsymbol{x_i} - \boldsymbol{\mu}_j) (\boldsymbol{x_i} - \boldsymbol{\mu}_j)^T \implies
$$

$$
\boldsymbol{\Sigma_j} = \frac{1}{N_j} \sum_{i=1}^N I[y_i = j] (\boldsymbol{x_i} - \boldsymbol{\mu}_j) (\boldsymbol{x_i} - \boldsymbol{\mu}_j)^T \tag{22}
$$

which is the MLE estimate of the covariance of the data points having class $$ j $$.

### Decision boundary

Now, having estimated the parameters of the assumed distributions, let's inspect the shape of the decision boundary. The decision boundary is the hypersurface separating data points from 2 different classes, let's say class $$ i $$ and class $$ j $$, so the points on this hypersurface have equal posterior probability of being assigned to either of the two classes: 

$$ \boldsymbol{x} \in \text{DecisionBoundary} \iff p(y = i \mid \boldsymbol{x}) = p(y = j \mid \boldsymbol{x}) \tag{23} $$

Using Bayes' rule as in (2), we can write an equivalent condition:

$$ 
\frac{p(\boldsymbol{x} \mid y = i) p(y = i)}{p(\boldsymbol{x})} = \frac{p(\boldsymbol{x} \mid y = j) p(y = j)}{p(\boldsymbol{x})} \iff
\frac{p(\boldsymbol{x} \mid y = i) p(y = i)}{p(\boldsymbol{x} \mid y = j) p(y = j)} = 1 \iff
$$

$$
\log \frac{p(\boldsymbol{x} \mid y = i)}{p(\boldsymbol{x} \mid y = j)} + \log \frac{p(y = i)}{p(y = j)} = 0 \iff
$$

$$

\log 
	\frac
	{
		\require{cancel} \cancel{\frac{1}{(2 \pi)^{\frac{D}{2}}}} 
		\frac{1}{\det(\boldsymbol{\Sigma_i})^{\frac{1}{2}}}
		e^{-\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu_i})^T \boldsymbol{\Sigma_i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_i})}
	}
	{
		\cancel{\frac{1}{(2 \pi)^{\frac{D}{2}}}}
		\frac{1}{\det(\boldsymbol{\Sigma_j})^{\frac{1}{2}}}
		e^{-\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu_j})^T \boldsymbol{\Sigma_j}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_j})}
	} + 
	\log \frac{\pi_i}{\pi_j}
= 0 \iff
$$

$$
\frac{1}{2} \log \frac{\det(\boldsymbol{\Sigma_j})}{\det(\boldsymbol{\Sigma_i})}
-\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu_i})^T \boldsymbol{\Sigma_i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_i})
+\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu_j})^T \boldsymbol{\Sigma_j}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_j})
+ \log \frac{\pi_i}{\pi_j} = 0 
$$

$$
\iff
$$

$$
\small
(\boldsymbol{x} - \boldsymbol{\mu_j})^T \boldsymbol{\Sigma_j}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_j}) -
(\boldsymbol{x} - \boldsymbol{\mu_i})^T \boldsymbol{\Sigma_i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu_i}) 
+ 2 \log \frac{\pi_i}{\pi_j}
+ \log \frac{\det(\boldsymbol{\Sigma_j})}{\det(\boldsymbol{\Sigma_i})} = 0 \tag{24}
$$

Now, looking at the expression above, we can see that the decision boundary is quadratic. We'll expand it further to see each coefficient clearly:

$$
\boldsymbol{x}^T \boldsymbol{\Sigma_j}^{-1} \boldsymbol{x} -
2 \boldsymbol{x}^T \boldsymbol{\Sigma_j}^{-1} \boldsymbol{\mu_j} +
\boldsymbol{\mu_j}^T \boldsymbol{\Sigma_j}^{-1} \boldsymbol{\mu_j} \\
- \boldsymbol{x}^T \boldsymbol{\Sigma_i}^{-1} \boldsymbol{x} +
2 \boldsymbol{x}^T \boldsymbol{\Sigma_i}^{-1} \boldsymbol{\mu_i} -
\boldsymbol{\mu_i}^T \boldsymbol{\Sigma_i}^{-1} \boldsymbol{\mu_i} + \\
2 \log \frac{\pi_i}{\pi_j}
+ \log \frac{\det(\boldsymbol{\Sigma_j})}{\det(\boldsymbol{\Sigma_i})} = 0 \iff
$$

$$
\boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x} + + \boldsymbol{x}^T  \boldsymbol{b} + c \tag{25}
$$

where

$$
\boldsymbol{A} = (\boldsymbol{\Sigma_j}^{-1} - \boldsymbol{\Sigma_i}^{-1}), \tag{26} 
$$

$$
\boldsymbol{b} = -2 (\boldsymbol{\Sigma_j}^{-1} \boldsymbol{\mu_j} - \boldsymbol{\Sigma_i}^{-1} \boldsymbol{\mu_i}), \tag{27}
$$

$$
c = \boldsymbol{\mu_j}^T \boldsymbol{\Sigma_j}^{-1} \boldsymbol{\mu_j} 
- \boldsymbol{\mu_i}^T \boldsymbol{\Sigma_i}^{-1} \boldsymbol{\mu_i}
+ 2 \log \frac{\pi_i}{\pi_j}
+ \log \frac{\det(\boldsymbol{\Sigma_i^{-1}})}{\det(\boldsymbol{\Sigma_j^{-1}})}  \tag{28}
$$

#### Linear Discriminant Analysis for classification

The model above might have too many parameters for some applications, so it can be simplified by assuming that the covariance matrix is the same for all classes, that is $$ \boldsymbol{\Sigma_i} = \boldsymbol{\Sigma}, \forall i \in \{1, 2, \dots k\} $$, in which case the decision boundary will be linear, because the $$ \boldsymbol{A} $$ term (26) will vanish.

### Code 

<!-- HTML generated using hilite.me --><div style="background: #f0f3f3; overflow:auto;width:auto;border:solid gray;border-width:.0em .0em .0em .0em;padding:.0em .0em;"><pre style="margin: 0; line-height: 125%"><span style="color: #006699; font-weight: bold">import</span> <span style="color: #00CCFF; font-weight: bold">numpy</span> <span style="color: #006699; font-weight: bold">as</span> <span style="color: #00CCFF; font-weight: bold">np</span>
<span style="color: #006699; font-weight: bold">from</span> <span style="color: #00CCFF; font-weight: bold">plotly</span> <span style="color: #006699; font-weight: bold">import</span> subplots
<span style="color: #006699; font-weight: bold">import</span> <span style="color: #00CCFF; font-weight: bold">plotly.graph_objs</span> <span style="color: #006699; font-weight: bold">as</span> <span style="color: #00CCFF; font-weight: bold">go</span>
<span style="color: #006699; font-weight: bold">import</span> <span style="color: #00CCFF; font-weight: bold">plotly.offline</span> <span style="color: #006699; font-weight: bold">as</span> <span style="color: #00CCFF; font-weight: bold">py</span>
<span style="color: #006699; font-weight: bold">from</span> <span style="color: #00CCFF; font-weight: bold">scipy.stats</span> <span style="color: #006699; font-weight: bold">import</span> multivariate_normal <span style="color: #006699; font-weight: bold">as</span> gaussian


<span style="color: #006699; font-weight: bold">class</span> <span style="color: #00AA88; font-weight: bold">GenerativeGaussianClassifier</span>(<span style="color: #336666">object</span>):
  <span style="color: #006699; font-weight: bold">def</span> <span style="color: #CC00FF">__init__</span>(<span style="color: #336666">self</span>, K):
    <span style="color: #336666">self</span><span style="color: #555555">.</span>K <span style="color: #555555">=</span> K

  <span style="color: #006699; font-weight: bold">def</span> <span style="color: #CC00FF">fit</span>(<span style="color: #336666">self</span>, Xs, ys):
    cls_counts <span style="color: #555555">=</span> np<span style="color: #555555">.</span>bincount(ys)
    get_mean <span style="color: #555555">=</span> <span style="color: #006699; font-weight: bold">lambda</span> k: Xs[ys <span style="color: #555555">==</span> k]<span style="color: #555555">.</span>mean(<span style="color: #FF6600">0</span>)
    get_cov <span style="color: #555555">=</span> <span style="color: #006699; font-weight: bold">lambda</span> k: np<span style="color: #555555">.</span>dot((Xs[ys <span style="color: #555555">==</span> k] <span style="color: #555555">-</span> <span style="color: #336666">self</span><span style="color: #555555">.</span>mus[k])<span style="color: #555555">.</span>T, \
    	                        Xs[ys <span style="color: #555555">==</span> k] <span style="color: #555555">-</span> <span style="color: #336666">self</span><span style="color: #555555">.</span>mus[k]) <span style="color: #555555">/</span> cls_counts[k]
    get_cls_cond <span style="color: #555555">=</span> <span style="color: #006699; font-weight: bold">lambda</span> k: gaussian(mean<span style="color: #555555">=</span><span style="color: #336666">self</span><span style="color: #555555">.</span>mus[k], cov<span style="color: #555555">=</span><span style="color: #336666">self</span><span style="color: #555555">.</span>gammas[k])

    <span style="color: #336666">self</span><span style="color: #555555">.</span>pis <span style="color: #555555">=</span> cls_counts <span style="color: #555555">/</span> <span style="color: #336666">len</span>(ys)
    <span style="color: #336666">self</span><span style="color: #555555">.</span>mus <span style="color: #555555">=</span> np<span style="color: #555555">.</span>stack([get_mean(k) <span style="color: #006699; font-weight: bold">for</span> k <span style="color: #000000; font-weight: bold">in</span> <span style="color: #336666">range</span>(<span style="color: #336666">self</span><span style="color: #555555">.</span>K)])
    <span style="color: #336666">self</span><span style="color: #555555">.</span>gammas <span style="color: #555555">=</span> np<span style="color: #555555">.</span>stack([get_cov(k) <span style="color: #006699; font-weight: bold">for</span> k <span style="color: #000000; font-weight: bold">in</span> <span style="color: #336666">range</span>(<span style="color: #336666">self</span><span style="color: #555555">.</span>K)])
    <span style="color: #336666">self</span><span style="color: #555555">.</span>cls_cond <span style="color: #555555">=</span> [get_cls_cond(k) <span style="color: #006699; font-weight: bold">for</span> k <span style="color: #000000; font-weight: bold">in</span> <span style="color: #336666">range</span>(<span style="color: #336666">self</span><span style="color: #555555">.</span>K)]
	
  <span style="color: #006699; font-weight: bold">def</span> <span style="color: #CC00FF">predict_scores</span>(<span style="color: #336666">self</span>, Xs):
    scores <span style="color: #555555">=</span> np<span style="color: #555555">.</span>zeros((<span style="color: #336666">len</span>(Xs), <span style="color: #336666">self</span><span style="color: #555555">.</span>K), <span style="color: #336666">float</span>)
    comp_score <span style="color: #555555">=</span> <span style="color: #006699; font-weight: bold">lambda</span> k, X: <span style="color: #336666">self</span><span style="color: #555555">.</span>pis[k] <span style="color: #555555">*</span> <span style="color: #336666">self</span><span style="color: #555555">.</span>cls_cond[k]<span style="color: #555555">.</span>pdf(X)
    <span style="color: #006699; font-weight: bold">for</span> i <span style="color: #000000; font-weight: bold">in</span> <span style="color: #336666">range</span>(<span style="color: #336666">len</span>(Xs)):
      scores[i] <span style="color: #555555">=</span> np<span style="color: #555555">.</span>stack([comp_score(k, Xs[i]) <span style="color: #006699; font-weight: bold">for</span> k <span style="color: #000000; font-weight: bold">in</span> <span style="color: #336666">range</span>(<span style="color: #336666">self</span><span style="color: #555555">.</span>K)])
    <span style="color: #006699; font-weight: bold">return</span> scores

  <span style="color: #006699; font-weight: bold">def</span> <span style="color: #CC00FF">predict</span>(<span style="color: #336666">self</span>, Xs):
    <span style="color: #006699; font-weight: bold">return</span> np<span style="color: #555555">.</span>argmax(<span style="color: #336666">self</span><span style="color: #555555">.</span>predict_scores(Xs), axis<span style="color: #555555">=</span><span style="color: #FF6600">1</span>)

  <span style="color: #006699; font-weight: bold">def</span> <span style="color: #CC00FF">accuracy</span>(<span style="color: #336666">self</span>, Xs, ys):
    <span style="color: #006699; font-weight: bold">return</span> np<span style="color: #555555">.</span>mean(<span style="color: #336666">self</span><span style="color: #555555">.</span>predict(Xs) <span style="color: #555555">==</span> ys)
</pre></div>


### Visualisation


![Probabilities](../img/generative_classification_gaussian.png)

#### Credits



