#!/usr/bin/env python
# coding: utf-8

# # Bayesian Inference
# 
# ## Bayesian Inference (Part-I Parametric)
# 
# ### Posterior
# 
# **Def.1 Posterior Distribution:** The posterior distribution of parameter $\theta$, based on observations $D=\{x_1,...,x_n\}$, likelihood function $L(\theta|D)=P(D|\theta)=\prod_{i=1}^n{P(x_i|\theta)}$(assuming data samples are i.i.d), prior distribution $\pi(\theta)$, is defined as:
# 
# \begin{align*}
# \pi(\theta|D)=\frac{P(D|\theta)\pi(\theta)}{P(D)}=\frac{P(D|\theta)\pi(\theta)}{\int_{\theta}{P(D|\theta)\pi(\theta)}}
# \end{align*}
# 
#  - To obtain the exact distribution(not the exact pdf) for $P(\theta|D)$, we could ignore the marginal probability $P(D)$ as constant(regarding $\theta$). Thus $\pi(\theta|D) \propto P(D|\theta)\pi(\theta)$. However, this case holds only if the form of such posterior resembles some well-known distributions, where we could find the kernel without considering the value of $P(D)$.
#  
# **Corollary.1-1 Joint Posterior:** The posterior in Def.1 can be applied to multiple parameters $\theta_1,...\theta_m$, if given the joint prior:
# 
# \begin{align*}
# \pi(\theta_1,...\theta_m|D)=\frac{P(D|\theta_1,...\theta_m)\pi(\theta_1,...\theta_m)}{P(D)}
# \end{align*}
# 
# **Corollary.1-2 Full Conditional Posterior:** For multiple parameters $\theta_1,...\theta_m$, the conditional posterior for i-th parameter $\theta_i$, if given values $\theta_{-i}$ for other parameters and data samples, is defined as:
# 
# \begin{align*}
# \pi(\theta_i|D,\theta_{-i})&=P(\theta_i|D,\theta_1,...,\theta_{i-1},\theta_{i+1},...\theta_m) \\
# &=\frac{\pi(\theta_1,...\theta_m|D)}{P(\theta_1,...,\theta_{i-1},\theta_{i+1},...\theta_m|D)} \\
# &=\frac{\pi(\theta_1,...\theta_m|D)}{\int_{\theta_i}{\pi(\theta_1,...\theta_m|D)\,d\theta_i}}
# \end{align*}
# 
#  - The full conditional posterior relies on computing the joint posterior first.
#  - In common practice, it is extremely complicated to compute the integral for each parameter $\theta_i$. One possible approach is to **derive the full conditional posteroir directly from the joint posterior**, by ignoring the other parameters as constants. Thus $P(\theta_i|D,\theta_{-i}) \propto P(\theta_1,...\theta_m|D)|_{\theta_{-i}}$ However, this case holds only for well-known distributions.
#  - The concept of full conditional is very important for Gibbs sampling method.
#  
# **Corollary.1-3 Marginal Posterior:** For multiple parameters $\theta_1,...\theta_m$, the marginal posterior for i-th parameter $\theta_i$, if given values $\theta_{-i}$ for other parameters and data samples, is defined as:
# 
# \begin{align*}
# \pi(\theta_i|D)&=\int_{\theta_{-i}}{\pi(\theta_1,...\theta_m|D)\,d\theta_{-i}} \\
# &=\int_{\theta_{-i}}{\pi(\theta_i,\theta_{-i}|D)\,d\theta_{-i}} \\
# &=\int_{\theta_{-i}}{\pi(\theta_i|D,\theta_{-i})P(\theta_{-i}|D)\,d\theta_{-i}} \\
# \end{align*}
# 
#  - The marginal posterior relies on computing the joint posterior first, or the full conditional posterior first.
#  - To obtain the exact distribution of marginal posterior, the integration might be simplified by ignoring some constants. Be aware that all parameters other than $\theta_i$ should be integrated out. However, the case holds only for well-known distributions.
# 
# ### Posterior Predictive
# 
# **Def.2 Posterior Predictive Distribution:** The posterior predictive distribution of parameter $\theta$ is defined as(based on conditions given in Def.1):
# 
# \begin{align*}
# P(x_{n+1}|D)=\int_{\theta}{P(x_{n+1},\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|D,\theta)P(\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|\theta)P(\theta|D)\,d\theta}
# \end{align*}
# 
#  - The posterior preditive relies on computing the posterior first.
# 

# ### Bayesian Prospective of MLP
# 
# In general, a machine learning problem could also be viewed from the bayesian inference prospective. The relationship could be described as:
# 
# | Bayesian Inference      | Machine Learning |
# | :-----------: | :-----------: |
# | Likelihood Function      | Loss Function       |
# | Prior Distribution   | Regularization        |
# | Posterior Distribution   | Regularized Loss Function       |
# | Maximize a Posterior Estimation(MAP)   | Minimize Regularized Loss Function       |
# | Maximize Likelihood Estimation(MLE)   | Minimize Loss Function       |

# ### Laplace Approximation
# 
# Intuitive: Approximate the posterior distribution with a multivariate normal distribution centered at the mode of posterior. 
# 
# **Def.3 Laplace Approximation:** Given a MAP estimation $\theta_{MAP}=argmax_{\theta}\pi(\theta|D)$, the laplace approximation of posterior on MAP is defined as:
# \begin{align*}
# & \ln{\pi(\theta|D)} \approx \ln{\pi(\theta_{MAP}|D)} - \frac{1}{2}(\theta-\theta_{MAP})^TA(\theta-\theta_{MAP}) \\
# & A = -\{\nabla_{\theta}^2\ln{\pi(\theta|D)}\}|_{\theta_{MAP}}
# \end{align*}
# In other words, the approximate k-dimensional multivariate normal distribution can be computed as:
# \begin{align*}
# & \pi(\theta|D) \sim \mathcal{N}_k(\mu, \Sigma), \theta \in \mathbb{R}^k \\
# & \mu=\theta_{MAP} \\
# & \Sigma=A^{-1}
# \end{align*}
# 
#  - Since $\ln{\pi(\theta_{MAP}|D)}$ is a constant, we could skip the computation. The kernel is already sufficiently defined by the 2nd term.
#  - Matrix $A$ is the minus of the posterior's Hessian at the MAP estimation. If dimension of parameter $\theta$ is k, then the dimension of $A$ is $k \times k$.
#  - The approximate k-dimensional multivariate normal distribution has the mean of $\theta_{MAP}$, which is basically how the method is designed, and the covariance matrix is the inverse of $A$. Considering the cost of computing inverse matrix is high($O(k^3)$), the dimension of parameters k should be limited.
#  

# #### Bayesian Information Criterion
# 
# Based on laplace approximation of the posterior distribution we have:
# 
# \begin{align*}
# & P(D) = \int_{\theta} P(D|\theta)\pi(\theta)\,d{\theta} \\
# & \pi(\theta|D) \propto P(D|\theta)\pi(\theta) \\
# & \pi(\theta|D) \approx \pi(\theta_{MAP}|D) exp\left(-\frac{1}{2}(\theta-\theta_{MAP})^TA(\theta-\theta_{MAP})\right) \\
# \end{align*}
# 
# Thus we could approximate the marginal distribution $P(D)$ as the following:
# 
# \begin{align*}
# P(D) & \approx \int_{\theta} \pi(\theta_{MAP}|D) exp\left(-\frac{1}{2}(\theta-\theta_{MAP})^TA(\theta-\theta_{MAP})\right) \,d{\theta} \\
# & \approx P(D|\theta_{MAP})\pi(\theta_{MAP}) \int_{\theta} exp\left(-\frac{1}{2}(\theta-\theta_{MAP})^TA(\theta-\theta_{MAP})\right) \,d{\theta} \\
# & = P(D|\theta_{MAP})\pi(\theta_{MAP}) (2\pi)^{\frac{k}{2}}det(A^{-1})^{\frac{1}{2}}
# \end{align*}
# 
# \begin{align*}
# \ln{P(D)} \approx \ln{P(D|\theta_{MAP})} + \ln\pi(\theta_{MAP}) + \frac{k}{2}\ln(2\pi) - \frac{1}{2}\ln(det(A))
# \end{align*}
# 
# When the sample size N is sufficiently large, $det(A)$ could be simplified as:
# 
# \begin{align*}
# det(A) \approx det(I(D, \theta_{MAP})) = N^k det(I_{unit}(D, \theta_{MAP}))
# \end{align*}
# 
# Based on such assumption, $P(D)$ could be further simplified as:
# 
# \begin{align*}
# \ln{P(D)} & \approx \ln{P(D|\theta_{MAP})} + \ln\pi(\theta_{MAP}) + \frac{k}{2}\ln(2\pi) - \frac{k\ln N}{2} - \frac{1}{2}\ln(det(I_{unit}(D, \theta_{MAP}))) \\
# & \approx \ln{P(D|\theta_{MAP})} - \frac{k\ln N}{2} \\
# \end{align*}
# 
# \begin{align*}
# P(D) \approx \exp\left(\ln P(D|\theta_{MAP}) -\frac{k\ln N}{2} \right) \approx exp(-\frac{BIC}{2})
# \end{align*}
# 
# Therefore, we derive the closed form for Bayesian Information Criterion(BIC):
# \begin{align*}
# BIC & = k\ln N - 2\ln P(D|\theta_{MLE}) \\
# & = k\ln N - 2\ln L(\theta_{MLE})
# \end{align*}
# 
#  - BIC, a.k.a Schwarz Information Criterion, is a metric for model selection based on finite training dataset with N data samples. Generally we prefer a model with smaller BIC.
#  - The intuitiion is to prevent overfitting by adding the penalty term for high dimensional parameters.

# ### Mean Field Approximation(Variational Inference)
# 
# Intuitive: Approximate the posterior distribution $\pi(\theta|D)$ with a fully factorized distribution $q(\theta)$. The basic idea is to treat each component of parameter $\theta_i$ as they are independent, such that the joint distribution is simple and tractable. Since the goal is to find the approximate distribution for each $\theta_i$, we are changing the form of such distributions during the training process as if they are variables, thus the method is also called Variational Inference.
# 
# #### Jensen's Inequality
# 
# **Def.4 Jensen's Inequality:** If X is a randome variable and $\varphi$ is a convex function on its domain, then we have:
# \begin{align*}
# \varphi(E[X]) \le E[\varphi(X)]
# \end{align*}
# 
#  - Def.4 could be proved by the properties of convex function. left $ = \varphi(\int{x}\,dP(x)) \le \int{\varphi(x)}\,dP(x) = $ right.
#  - Intuitively, we could view the integration as sum(with infinite terms), thus the real value(left) is upper bounded by the interpolation(right), because the interpolation of convex function is always over-estimated.
#  
# #### Kullback–Leibler Divergence
# 
# **Def.5 KL Divergence:** A method to measure the distance between two distributions P & Q on the same random variable X, is defined as:
# \begin{align*}
# D_{KL}(P||Q) = \int_{x}{P(x)\ln{\frac{P(x)}{Q(x)}}}\,dx
# \end{align*}
# 
#  - There are other choices to measure the difference, to be continued...
# 
# #### Mean Field Approximation
# 
# For simplicity, we let $q(\theta)$ be a fully factorized distribution, such that:
# \begin{align*}
# q(\theta) = \prod_{i=1}^m{q_i(\theta_i)}
# \end{align*}
# 
# The goal is to minimize the KL divergence between $q(\theta)$ and $\pi(\theta|D)$, so that the approximate distribution is a better estimation of the posterior. First we derive a lower bound of $\ln{P(D)}$:
# 
# \begin{align*}
# \ln{P(D)} & = \ln \int_{\theta}{P(D,\theta)}\,d\theta = \ln_{\theta} \int_{\theta} q(\theta)\frac{P(D,\theta)}{q(\theta)} \,d\theta \\
# & = \ln E_{q(\theta)}[\frac{P(D,\theta)}{q(\theta)}] \\
# & \ge E_{q(\theta)}[\ln\frac{P(D,\theta)}{q(\theta)}] \\
# & = \int_{\theta} q(\theta) \ln\frac{P(D,\theta)}{q(\theta)} \,d\theta \\
# & = \int_{\theta} q(\theta) \ln\frac{\pi(\theta|D)P(D)}{q(\theta)} \,d\theta \\
# & = \ln P(D) + \int_{\theta} q(\theta) \ln\frac{\pi(\theta|D)}{q(\theta)} \,d\theta \\
# & = \ln P(D) - D_{KL}(q(\theta) || \pi(\theta|D))
# \end{align*}
# 
# We denote the lower bound as $L(q)$, where distribution $q$ is the variable in the training process. Now recall that the distribution $q$ can be fully factorized:
# \begin{align*}
# L(q) &= \int_{\theta} q(\theta) \ln\frac{P(D,\theta)}{q(\theta)} \,d\theta \\
# &= \int_{\theta} q(\theta) \ln P(D,\theta) \,d\theta - \int_{\theta} q(\theta) \ln q(\theta) \,d\theta \\
# &= \int_{\theta_i} q_i(\theta_i) \left\{\int_{\theta_{j \neq i}} \ln P(D,\theta) \prod_{j \neq i}{q_j(\theta_{j})} \,d{\theta_{j \neq i}} \right\} \,d\theta_i - \sum_{j}{\int_{\theta_j} q_j(\theta_j) \ln q_j(\theta_j) \,d\theta_j} \\
# &= \int_{\theta_i} q_i(\theta_i) E_{q(\theta_{j \neq i})}[\ln P(D,\theta)] \,d{\theta_i} + const
# \end{align*}
# 
# We denote the distribution un-related to $\theta_i$ as $\ln P_{j \neq i}(D,\theta)= E_{q(\theta_{j \neq i})}[\ln P(D,\theta)] + const$. If we only focus on terms related to the i-th component $q_i(\theta_i)$:
# \begin{align*}
# L_i(q_i) &= \int_{\theta_i} q_i(\theta_i) \ln P_{j \neq i}(D,\theta) \,d{\theta_i} - \int_{\theta_i} q_i(\theta_i) \ln q_i(\theta_i) \,d\theta_i \\
# &= \int_{\theta_i} q_i(\theta_i) [\ln \frac{P_{j \neq i}(D,\theta)}{q_i(\theta_i)}] \,d{\theta_i} \\
# &= - D_{KL} (q_i(\theta_i)||P_{j \neq i})
# \end{align*}
# 
# **Def.6 Mean Field Appoximation:** if we minimize the KL divergence as shown above, distribution $q_i(\theta_i)$ should have the form of:
# \begin{align*}
# q_i(\theta_i) &= P_{j \neq i}(D,\theta) \\
# &= exp\{E_{q(\theta_{j \neq i})}[\ln P(D,\theta)] + const\} \\
# &\propto exp\{E_{q(\theta_{j \neq i})}[\ln P(D,\theta)]\}
# \end{align*}
# 
# \begin{align*}
# q_i(\theta_i) = \frac{exp\{E_{q(\theta_{j \neq i})}[\ln P(D,\theta)]\}}{\int_{\theta_i} exp\{E_{q(\theta_{j \neq i})}[\ln P(D,\theta)]\} \,d\theta_i}
# \end{align*}
# 
#  - The key concept for mean-field is that $q_i \propto E_{q(\theta_{j \neq i})}[\ln P(D,\theta)]$, where all parameters other than $\theta_i$ will be integrated out by computing the expectation. The denominator is a normalizing constant, and can be ignored if $q_i$ takes the form of well-known distributions.
# 

# ## Bayesian Inference (Part-II Non-Parametric)
# 
# ### Gaussian Process
# 
# #### Multivariate Normal Distribution
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
# #### Covariance Function(Kernel Function)
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
# 
# #### Gaussian Process for Regression (GPR)
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
# 

# ### Dirichlet Process
# 
# Intuition: A general methodology for clustering, with unlimited number of clusters.
# 
# #### Multinomial Distribution
# 
# Multinomial distribution is the generalization of binomial distribution to higher dimension. Suppose we have some observations distributed to $K$ categories $p = (p_1,p_2,...p_K)$ given $\sum_{i=1}^K p_i = 1$, the number of occurence for category $i$ is denoted as $n_i$, thus the total observations $n = \sum_{i=1}^K n_i$. For instance, the experiment resembles tossing a K-side coin n times.
# 
# Probability mass function(pdf in discrete form): giving $\Gamma(n) = (n-1)!$
# \begin{align*}
# P(n_1,n_2,...,n_K \mid p_1,p_2,...p_K) &= \frac{n!}{n_1!n_2!...n_K!} p_1^{n_1}p_2^{n_2}...p_K^{n_K} \\
# & = \frac{\Gamma(n+1)}{\prod_{i=1}^K \Gamma(n_i+1)} \prod_{i=1}^K p_i^{n_i}
# \end{align*}
# 
# #### Beta Distribution
# 
# Beta Distribution is a continuous distribution defined on $[0, 1]$, with two positive parameters $\alpha$ and $\beta$. The distribution is widely used as the prior for Bernoulli, (Negative)Binomial and Geometric distributed data because of conjugacy. 
# 
# Probability density function: 
# \begin{align*}
# P(x \mid \alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1}(1-x)^{\beta-1}
# \end{align*}
# 
# #### Dirichlet Distribution
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
#  
# #### Dirichlet Process 
# 
# **Def. Dirichlet Process:** Let $S$ be a measuable set, $\alpha$ is a positive real number(NOT vector), and $G$ is a base distribution. We define Dirichlet Process(DP) with parameter $\alpha$ and $G$ as the random measure, where with any possible partition of $S = \{S_1,S_1,...S_K\}$, it would follow a dirichlet distribution with parameter $Dir(\alpha G(S_1),\alpha G(S_2),...,\alpha G(S_K))$.
# 
# Comments:
#  - DP is the infinite-dimensional generalization of dirichlet distribution.
#  - By definition, we could see that the output of DP is a new distribution.
#  - Parameter $G$ is the base, or "mean" of distributions generated by DP. In other words, DP draws various distribution around $G$, and $G$ is the expected value of the process.
#  - Parameter $\alpha$ is called the concentration parameter on two aspects:
#      - Concentration: describe how close the distributions are drawn around $G$.
#      - Discretization: describe how discrete the output distributions are.
#      
#   | $\alpha$ value   | Concentration      | Discretization |    Expectation |
#   | :-----------: | :-----------: | :-----------: |   :-----------: |   
#   | small   |  far away from $G$  | discrete, degenerate to point masses |  highly influenced by training data | 
#   | large   | close around $G$    | (nearly) continuous  |  highly influenced by the prior  |
#   
# Properties: *Conjugacy of DP*
# 
# 1. Set the prior distribution on $G \sim DP(\alpha, G_0)$, for instance, the base distribution $G_0$ is a normal distribution $\mathbb{N}(\mu_0, \sigma_0^2)$.
# 2. Collect observations $P(x_1, x_2, ..., x_n) \sim G$, i.i.d.
# 3. Then the updated posterior distribution follows DP:
# \begin{align*}
# G|x_1, x_2, ..., x_n \sim DP(\alpha + n, \frac{\alpha G_0 + \sum_{i=1}^{n} \delta_{x_i}}{\alpha + n})
# \end{align*}
# 
# Properties: *Samples of DP*
# 
# Suppose we draw some data samples $(y_1, y_2, ..., y_n) \sim DP(\alpha, G_0)$, the values have the properties of following:
#  - For a new data sample $y_k$, the probability of such data sample is driven from follows:
#  \begin{align*}
#  y_k  \sim \left\{
#  \begin{array}{ll}
#      y_i & w.p. \frac{1}{\alpha+k-1}  \\
#      G_0 & w.p. \frac{\alpha}{\alpha+k-1} \\
#  \end{array}
#  \right.
#  \end{align*}
#  - Data samples $(y_1, y_2, ..., y_n)$ are NOT independent, but exchangeable. For instance, when generating the new data sample $y_k$, it relies on the distribution of previous samples $(y_1,y_2,...,y_{k-1})$, but at the same time  the ordering of these samples is ignored. In other words, given a certain distribution $DP(\alpha, G_0)$, the data samples are *conditionally independent*.
# 
# 
# #### Chinese Restaurant Process(CPR)
# Intuition: A method of generating data samples whose clusters(partitions) follows a DP process.
# 
# To generate $DP(\alpha, G_0)$, for the k-th data sample, it has two options:
#  - start a new table with a fixed and small probability;
#  - join a existed table(created by the previous data samples);
# 
#  \begin{align*}
#  x_k  = \left\{
#  \begin{array}{ll}
#      val(cluster_i) & w.p. \frac{count(cluster_i)}{\alpha+k-1}  \\
#      sample(G_0) & w.p. \frac{\alpha}{\alpha+k-1} \\
#  \end{array}
#  \right.
#  \end{align*}
# 
# Comments: 
#  - Since the clusters(partitions) of CPR follows DP, they shall have the properties of exchangeability as explained in previous chapater.
#  - Given that the probability of joining a existed cluster is propotional to its size(number of data samples), it is obvious that the new data sample would have a higher chance of joining the largest cluster, a.k.a. "Rich got Richer".
#  - For a certain dataset with $(x_1, x_2, ...,x_n)$, the probability of having exactly "k" clusters is:
#  \begin{align*}
#      P(cnt(clusters)=k | x_1, x_2, ...,x_n) = \frac{\alpha^k}{\alpha(\alpha+1)\cdots(\alpha+n-1)} \sum_{i=1}^k (size(cluster_i) - 1)!
#  \end{align*}
# 
