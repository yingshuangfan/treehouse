#!/usr/bin/env python
# coding: utf-8

# # Bayesian Inference
# 
# ## Posterior
# 
# **Def.1 Posterior Distribution:** The posterior distribution of parameter $\theta$, based on observations $D=\{x_1,...,x_n\}$, likelihood function $L(\theta|D)=P(D|\theta)=\prod_{i=1}^n{P(x_i|\theta)}$(assuming data samples are i.i.d), prior distribution $\pi(\theta)$, is defined as:
# 
# \begin{align}
# \pi(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}=\frac{P(D|\theta)\pi(\theta)}{\int_{\theta}{P(D|\theta)\pi(\theta)}}
# \end{align}
# 
#  - To obtain the exact distribution(not the exact pdf) for $P(\theta|D)$, we could ignore the marginal probability $P(D)$ as constant(regarding $\theta$). Thus $P(\theta|D) \propto P(D|\theta)P(\theta)$. However, this case holds only if the form of such posterior resembles some well-known distributions, where we could find the kernel without considering the value of $P(D)$.
#  
# **Corollary.1-1 Joint Posterior:** The posterior in Def.1 can be applied to multiple parameters $\theta_1,...\theta_m$, if given the joint prior:
# 
# \begin{align}
# \pi(\theta_1,...\theta_m|D)=\frac{P(D|\theta_1,...\theta_m)\pi(\theta_1,...\theta_m)}{P(D)}
# \end{align}
# 
# **Corollary.1-2 Full Conditional Posterior:** For multiple parameters $\theta_1,...\theta_m$, the conditional posterior for i-th parameter $\theta_i$, if given values $\theta_{-i}$ for other parameters and data samples, is defined as:
# 
# \begin{align}
# \pi(\theta_i|D,\theta_{-i})&=P(\theta_i|D,\theta_1,...,\theta_{i-1},\theta_{i+1},...\theta_m) \\
# &=\frac{\pi(\theta_1,...\theta_m|D)}{P(\theta_1,...,\theta_{i-1},\theta_{i+1},...\theta_m|D)} \\
# &=\frac{\pi(\theta_1,...\theta_m|D)}{\int_{\theta_i}{\pi(\theta_1,...\theta_m|D)\,d\theta_i}}
# \end{align}
# 
#  - The full conditional posterior relies on computing the joint posterior first.
#  - In common practice, it is extremely complicated to compute the integral for each parameter $\theta_i$. One possible approach is to **derive the full conditional posteroir directly from the joint posterior**, by ignoring the other parameters as constants. Thus $P(\theta_i|D,\theta_{-i}) \propto P(\theta_1,...\theta_m|D)|_{\theta_{-i}}$ However, this case holds only for well-known distributions.
#  - The concept of full conditional is very important for Gibbs sampling method.
#  
# **Corollary.1-3 Marginal Posterior:** For multiple parameters $\theta_1,...\theta_m$, the marginal posterior for i-th parameter $\theta_i$, if given values $\theta_{-i}$ for other parameters and data samples, is defined as:
# 
# \begin{align}
# \pi(\theta_i|D)&=\int_{\theta_{-i}}{\pi(\theta_1,...\theta_m|D)\,d\theta_{-i}} \\
# &=\int_{\theta_{-i}}{\pi(\theta_i,\theta_{-i}|D)\,d\theta_{-i}} \\
# &=\int_{\theta_{-i}}{\pi(\theta_i|D,\theta_{-i})P(\theta_{-i}|D)\,d\theta_{-i}} \\
# \end{align}
# 
#  - The marginal posterior relies on computing the joint posterior first, or the full conditional posterior first.
#  - To obtain the exact distribution of marginal posterior, the integration might be simplified by ignoring some constants. Be aware that all parameters other than $\theta_i$ should be integrated out. However, the case holds only for well-known distributions.
# 
# ## Posterior Predictive
# 
# **Def.2 Posterior Predictive Distribution:** The posterior predictive distribution of parameter $\theta$ is defined as(based on conditions given in Def.1):
# 
# \begin{align}
# P(x_{n+1}|D)=\int_{\theta}{P(x_{n+1},\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|D,\theta)P(\theta|D)\,d\theta}=\int_{\theta}{P(x_{n+1}|\theta)P(\theta|D)\,d\theta}
# \end{align}
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
