#!/usr/bin/env python
# coding: utf-8

# # Bayesian Inference
# 
# ## Posterior
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
# ## Posterior Predictive
# 
# **Def.2 Posterior Predictive Distribution:** The posterior predictive distribution of parameter $\theta$ is defined as(based on conditions given in Def.1):
# 
# \begin{align*}
# P(x_{n+1}|D)=\int_{\theta}{P(x_{n+1},\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|D,\theta)P(\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|\theta)P(\theta|D)\,d\theta}
# \end{align*}
# 
#  - The posterior preditive relies on computing the posterior first.
# 

# ## Bayesian Prospective of MLP
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

# ## Laplace Approximation
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

# ### Multivariate Normal Distribution
# 
# Multivariate normal distribution is the generalization of normal distribution to higher dimension. Suppose the dimension is k, then we denote the k dimension multivariate normal distribuion as $\mathcal{N}_k(\mu, \Sigma)$, where $\Sigma$ is semi-positive definite. In addition, the distribution is non-degenerate, if $\Sigma$ is positive definite. For simplicity we assume that a distribution is always non-degenerate.
# 
# Properties:
#  - mean: $\mu$
#  - mode: $\mu$
#  - variance: $\Sigma$
# 
# Probability density function(non-degenerate):
# \begin{align*}
# f(X) = \frac{exp\left(-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)}{(2\pi)^{\frac{k}{2}}det(\Sigma)^{\frac{1}{2}}}
# \end{align*}
# 

# ### Bayesian Information Criterion
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

# ## Mean Field Approximation(Variational Inference)
# 
# Intuitive: Approximate the posterior distribution $\pi(\theta|D)$ with a fully factorized distribution $q(\theta)$. The basic idea is to treat each component of parameter $\theta_i$ as they are independent, such that the joint distribution is simple and tractable. Since the goal is to find the approximate distribution for each $\theta_i$, we are changing the form of such distributions during the training process as if they are variables, thus the method is also called Variational Inference.
# 
# ### Jensen's Inequality
# 
# **Def.4 Jensen's Inequality:** If X is a randome variable and $\varphi$ is a convex function on its domain, then we have:
# \begin{align*}
# \varphi(E[X]) \le E[\varphi(X)]
# \end{align*}
# 
#  - Def.4 could be proved by the properties of convex function. left $ = \varphi(\int{x}\,dP(x)) \le \int{\varphi(x)}\,dP(x) = $ right.
#  - Intuitively, we could view the integration as sum(with infinite terms), thus the real value(left) is upper bounded by the interpolation(right), because the interpolation of convex function is always over-estimated.
#  
# ### Kullback–Leibler Divergence
# 
# **Def.5 KL Divergence:** A method to measure the distance between two distributions P & Q on the same random variable X, is defined as:
# \begin{align*}
# D_{KL}(P||Q) = \int_{x}{P(x)\ln{\frac{P(x)}{Q(x)}}}\,dx
# \end{align*}
# 
#  - There are other choices to measure the difference, to be continued...
# 
# ### Mean Field Approximation
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
