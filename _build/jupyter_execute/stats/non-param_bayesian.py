#!/usr/bin/env python
# coding: utf-8

# # Bayesian Inference (Part-II Non-Parametric)
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

# ## Dirichlet Process
# 
# Intuition: A general methodology for clustering, with unlimited number of clusters.
# 
# ### Multinomial Distribution
# 
# Multinomial distribution is the generalization of binomial distribution to higher dimension. Suppose we have some observations distributed to $K$ categories $p = (p_1,p_2,...p_K)$ given $\sum_{i=1}^K p_i = 1$, the number of occurence for category $i$ is denoted as $n_i$, thus the total observations $n = \sum_{i=1}^K n_i$. For instance, the experiment resembles tossing a K-side coin n times.
# 
# Probability mass function(pdf in discrete form): giving $\Gamma(n) = (n-1)!$
# \begin{align*}
# P(n_1,n_2,...,n_K \mid p_1,p_2,...p_K) &= \frac{n!}{n_1!n_2!...n_K!} p_1^{n_1}p_2^{n_2}...p_K^{n_K} \\
# & = \frac{\Gamma(n+1)}{\prod_{i=1}^K \Gamma(n_i+1)} \prod_{i=1}^K p_i^{n_i}
# \end{align*}
# 
# ### Beta Distribution
# 
# Beta Distribution is a continuous distribution defined on $[0, 1]$, with two positive parameters $\alpha$ and $\beta$. The distribution is widely used as the prior for Bernoulli, (Negative)Binomial and Geometric distributed data because of conjugacy. 
# 
# Probability density function: 
# \begin{align*}
# P(x \mid \alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1}(1-x)^{\beta-1}
# \end{align*}
# 
# ### Dirichlet Distribution
# 
# Dirichlet distribution is the generalization of beta distribution to higher dimension. Likewise, suppose we have some observations distributed to $K$ categorites, the concentration parameters is defined as $Dir (\alpha_1,\alpha_2,...\alpha_K)$. Note that the sum of $\alpha_i$ is not required to be 1, e.g. $Dir(0.1,0.1)$ is a valid distribution. 
# 
# Probability density function: 
# \begin{align*}
# P(x_1,x_2,...,x_k \mid \alpha_1,\alpha_2,...\alpha_K) = \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i-1}
# \end{align*}
# 
# Similarly, dirichlet distribution is conjugate over multinomial distributed data. If we define the prior with a dirichlet distribution $p = (p_1,p_2,...,p_K) \sim Dir(\alpha_1,\alpha_2,...,\alpha_K)$, and the observed multinomial distributed data is denoted as $(n_1,n_2,...,n_K)$, then the likelihood is just the probability mass function of multinomial distribution:
# \begin{align*}
# P(n_1,n_2,...,n_K \mid p_1,p_2,...p_K) = \frac{\Gamma(n+1)}{\prod_{i=1}^K \Gamma(n_i+1)} \prod_{i=1}^K p_i^{n_i}
# \end{align*}
# 
# And the updated posterior is also dirichlet distribution(because of conjugacy):
# \begin{align*}
# \pi(p_1,p_2,...p_K \mid n_1,n_2,...,n_K) &\propto P(n_1,n_2,...,n_K \mid p_1,p_2,...p_K)\pi(p_1,p_2,...p_K) \\
# &\propto \frac{\Gamma(n+1)}{\prod_{i=1}^K \Gamma(n_i+1)} \prod_{i=1}^K p_i^{n_i} \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K \Gamma(\alpha_i)} \prod_{i=1}^K p_i^{\alpha_i-1} \\
# &\propto \prod_{i=1}^K p_i^{\alpha_i+n_i-1} \\
# &\sim Dir(\alpha_1+n_1,\alpha_2+n_2,...,\alpha_K+n_K)
# \end{align*}
# 
# Comments: 
#  - We could see that the update of posterior with giving training data is quite straight forward: add-in the number of occurence for each category on the prior.
#  
# Properties:
#  ... to be continued.

# ### Dirichlet Process 
# 
# **Def. Dirichlet Process:** Let $S$ be a measuable set, $\alpha$ is a positive real number(NOT vector), and $G$ is a base distribution. We define Dirichlet Process(DP) with parameter $\alpha$ and $G$ as the random measure, where with any possible partition of $S = \{S_1,S_1,...S_K\}$, it would follow a dirichlet distribution with parameter $Dir(\alpha G(S_1),\alpha G(S_2),...,\alpha G(S_K))$.
# 
# Comments:
#  - DP is the infinite-dimensional generalization of dirichlet distribution.
#  - By definition, we could see that the output of DP is a new distribution.
#  - Parameter $G$ is the base, or "mean" of distributions generated by DP. In other words, DP draws various distribution around $G$, and $G$ is the expected value of the process.
#  - Parameter $\alpha$ is called the concentration parameter on two aspects:
#      - Concentration: describe how close the distributions are drawn aroun $G$.
#      - Discretization: describe how discrete the output distributions are.
#      
#   | $\alpha$ value   | Concentration      | Discretization |    Expectation |
#   | :-----------: | :-----------: | :-----------: |   :-----------: |   
#   | small   |  far away from $G$  | discrete, degenerate to point masses |  highly influenced by training data | 
#   | large   | close around $G$    | (nearly) continuous  |  highly influenced by the prior  |
