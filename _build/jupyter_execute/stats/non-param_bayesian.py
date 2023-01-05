#!/usr/bin/env python
# coding: utf-8

# # Non-Parametric Bayesian
# 
# ## Gaussian Process
# 
# ### Multivariate Normal Distribution
# 
# Multivariate normal distribution is the generalization of normal distribution to higher dimension. Suppose the dimension is $d$, then we denote the $d$ dimension multivariate normal distribuion as $\mathcal{N}_d(\mu, \Sigma)$, where $\Sigma$ is semi-positive definite. In addition, the distribution is non-degenerate, if $\Sigma$ is positive definite. For simplicity we assume that a distribution is always non-degenerate.
# 
# **Def. Multivariate Normal Distribution:** $X \sim \mathcal{N}_d(\mu, \Sigma)$, with mean vector $\mu \in \mathbb{R}^d$ and covariance matrix $\mu \in \mathbb{R}^{d \times d}$
# 
# Properties:
#  - mean: $\mu$
#  - mode: $\mu$
#  - variance: $\Sigma$
# 
# Probability density function(non-degenerate):
# \begin{align*}
# f(X) = \frac{exp\left(-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)}{(2\pi)^{\frac{d}{2}}det(\Sigma)^{\frac{1}{2}}}
# \end{align*}
# 
# Comments: 
#  - If we fit training data with a multivariate normal distribution model, then the parameters are mean vector and covariance matrix as shown in the definition. In other words, if we want to generalize the model into unlimited(infinite) number of parameters, ont possible approach is to extend the dimension $d$ to infinite, thus we introduce the method of Gaussian Process.
#  
# **Def. Gaussian Process:** The Gaussian process is a stochastic process, a collection of random variables $\{x_1, x_2,...\}$ indexed by time or space, such that every finite collection of these random variables follows a multivariate normal distribution. We denote $y = \{y(x), x \in \mathcal{X}\} \sim GP(m(\cdot), k(\cdot,\cdot))$, with mean function $m(\cdot)$ and covariance function $k(\cdot,\cdot))$. By definition, any finite subset of $X = \{x_1,x_2,...x_n\}$ should follow a multivariate distribution $y(X) \sim \mathcal{N}(m(X), k(X, X))$.
# 
# Comments:
#  - Compared to other parametric model, GP is different because it doesn't output new data point(prediction) directly, instead it outputs a function that fits the data. Thus, the GP is also called **the distribution over functions**.
#  - Given a certain training data, the fitted GP model could produce infinite numbers of possible function that fit the training data, thus the parameters are unlimited, or so-called non-parametric. The idea make sense, because there are unlimited ways to fit a certain dataset in general.
#  
# ### Covariance Function(Kernel Function)
# 
# Gaussian Process is defined by a mean function $m(\cdot)$ and a covariance function(a.k.a kernel) $k(\cdot,\cdot))$. For simplicity, the mean function can be ignored if we assumed the training data is already scaled to mean zero. Thus the main focus of GP is the choice of covariance function. In other words, the covariance function completely defines the GP's behavior.
# 
# Properties:
#  - Stationary: $k(X, X')$ depends only on $X - X'$.
#  - Isotropic: $k(X, X')$ depends only on $|X - X'|$. (Ignore the directions)
#  
# Common choices: 
#  - Constant:
# \begin{align*}
# k(X, X') = C
# \end{align*}
#  - Linear:
# \begin{align*}
# k(X, X') = X^TX'
# \end{align*}
#  - Squared Exponential(RBF):
# \begin{align*}
# k(X, X') = exp\left(-\frac{|X-X'|^2}{2l^2} \right)
# \end{align*}
#  - Ornstein-Uhlenbeck:
# \begin{align*}
# k(X, X') = exp\left(-\frac{|X-X'|}{l} \right)
# \end{align*}
#  - White Noice: 
# \begin{align*}
# &k(X, X') = \sigma^2 \delta_{X,X'} \\
# &\delta_{X,X'} = \left\{
# \begin{array}{ll}
#       1 & X \neq X' \\
#       0 & X = X' \\
# \end{array} 
# \right. 
# \end{align*}
# 
# Comments: The covariance function clearly resembles a measure of distance between two dataset $X$ and $X'$, it describes our prior knowledge on the correlations bewteen observed data points, or in other words, their similarites.

# ### Gaussian Process for Regression (GPR)
# 
# Intuition: We choose Gaussian Process as prior and define the covariance function, and update the GP with training data to generate a posterior. 
# 
# Suppose the prior $f \sim GP(m(\cdot), k(\cdot, \cdot))$. We denote the observed data as $X \in \mathbb{R}^n$, and the goal is to predict future observations $X^* \in \mathbb{R}^{n^*}$. By defintion, the combination of these data points should follow a multivariate normal distribution:
# 
# \begin{align*}
# \begin{bmatrix}
# f(X)\\
# f(X^*)
# \end{bmatrix}
# \sim \mathcal{N}_{n+n^*}\left(
# \begin{bmatrix}
# \mu_1\\
# \mu_2
# \end{bmatrix}
# ,
# \begin{bmatrix}
# \Sigma_{11} & \Sigma_{12}\\
# \Sigma_{21} & \Sigma_{22}
# \end{bmatrix}
# \right)\\
# \end{align*}
# 
# \begin{align*}
# &\mu_1 = m(X)\\
# &\mu_2 = m(X^*)\\
# &\Sigma_{11} = k(X,X)\\
# &\Sigma_{12} = k(X,X^*)\\
# &\Sigma_{21} = k(X^*,X)\\
# &\Sigma_{22} = k(X^*,X^*)
# \end{align*}
# 
# Note that $\Sigma_{12} = \Sigma_{21}^T$.
# 
# The posterior predictive $f(X^*)|f(X)$ could be viewed as the conditional distribution of the multivariate normal distribution:
# \begin{align*}
# f(X^*)|f(X) \sim \mathcal{N}_{n^*}(\mu_{2|1}, \Sigma_{2|1})
# \end{align*}
# \begin{align*}
# \mu_{2|1} &= \mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(f(x) - \mu_1) \\
# &= m(X^*) + k(X^*,X)k(X,X)^{-1}(f(x) - m(X)) \\
# \end{align*}
# \begin{align*}
# \Sigma_{2|1} &= \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12} \\
# &= k(X^*, X^*) - k(X^*,X)k(X,X)^{-1}k(X,X^*)
# \end{align*}
# 
# 
# **Example. GPR with white-noised data:** If the observed data contains noise $y_i = f(x_i) + \sigma_i$, where the noise is white noise $\sigma_i \sim \mathcal{N}(0, \sigma^2)$ with i.i.d. Then the joint distribution for current observations $y$ and future observations $y^*$ is:
# 
# \begin{align*}
# \begin{bmatrix}
# y\\
# y^*
# \end{bmatrix}
# \sim \mathcal{N}_{n+n^*}\left(
# \begin{bmatrix}
# \mu_1'\\
# \mu_2'
# \end{bmatrix}
# ,
# \begin{bmatrix}
# \Sigma_{11}' & \Sigma_{12}'\\
# \Sigma_{21}' & \Sigma_{22}'
# \end{bmatrix}
# \right)\\
# \end{align*}
# 
# \begin{align*}
# &\mu_1' = m(X)\\
# &\mu_2' = m(X^*)\\
# &\Sigma_{11}' = k(X,X)+\sigma^2I_n\\
# &\Sigma_{12}' = k(X,X^*)\\
# &\Sigma_{21}' = k(X^*,X)\\
# &\Sigma_{22}' = k(X^*, X^*)+\sigma^2I_{n^*}
# \end{align*}
# 
# The posterior predictive $y^*|y$ is:
# \begin{align*}
# y^*|y \sim \mathcal{N}_{n^*}(\mu_{2|1}', \Sigma_{2|1}')
# \end{align*}
# \begin{align*}
# \mu_{2|1}' &= \mu_2' + \Sigma_{21}'\Sigma_{11}'^{-1}(y - \mu_1') \\
# &= m(X^*) + k(X^*,X)(k(X,X)+\sigma^2I_n)^{-1}(y - m(X)) \\
# \end{align*}
# \begin{align*}
# \Sigma_{2|1}' &= \Sigma_{22}' - \Sigma_{21}'\Sigma_{11}'^{-1}\Sigma_{12}' \\
# &= (k(X^*, X^*)+\sigma^2I_{n^*}) - k(X^*,X)(k(X,X)+\sigma^2I_n)^{-1}k(X,X^*)
# \end{align*}
