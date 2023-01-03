#!/usr/bin/env python
# coding: utf-8

# # Sampling
# 
# ## Direct Sampling
# 
# ## Likelihood(weighted) Sampling

# ## Rejection Sampling
# 
# 

# ## Importance Sampling
# 

# ## Markov Chain Monte Carlo(MCMC)
# 

# ## Gibbs Sampler

# ## Monte Carlo Estimation (Application of Sampling)
# 
# Intuition: Estimatation based on random sampling, a.k.a Monte Carlo Method. The methodology includes various implementations, a common adoption in statistics is estimating the integration(e.g. marginal probability) with complicated closed-form expression.
# 
# **Def. Monte-Carlo Estimation:** The approximate expectation of $f(X)$, where random variable X follows distribution $P$, is the average of i.i.d sampled data $z_1,z_2,...,z_N$. 
# 
# \begin{align*}
# E_{x \sim p(x)}[f(x)] = \int p(x)f(x) \,dx \approx \frac{1}{N} \sum_{i=1}^N f(z_i)
# \end{align*}
# 
# **Corollary. Estimation of Posterior Predictive with Monte Carlo Method:** The posterior predictive (integration of posterior over possible parameters $\theta$) can be estimated by:
# 
# \begin{align*}
# P(x_{n+1}|D) &=\int_{\theta}{P(x_{n+1}|\theta)P(\theta|D)\,d\theta} \\
# & \approx \frac{1}{N} \sum_{i=1}^N P(x_{n+1}|\theta_i) \\
# & \theta_i \sim P(\theta|D)
# \end{align*}
# 
# The estimation can be described as: 
# 1. Draw a sampled parameter $\theta_i$ from the posterior $P(\theta|D)$ with a sampler(e.g. Gibbs Sampler). 
# 2. Given the $\theta_i$ in last step, draw a sampled new data $z_i$ from the likelihood $P(x_{n+1}|\theta_i)$.
# 3. Repeat step 1&2 in order to generate the i.i.d sampled data $z_1,z_2,...,z_N$, giving the size of such dataset(N) is sufficiently large, then the average of sampled data is an Monte Carlo Estimation of new data point $x_{new}$.
# 
# Comments: 
#  - Notice that we only sample exactly ONE $z_i$ from any given $\theta_i$ sampled from the posterior. The point here is that the size of $z_i$ doesn't matter, as long as we have sufficent samples of $\theta_i$. In other words, the estimation here relies on taking the "average" over $\,d\theta$. For instance, a sampled data of 10k with 5k $\theta$ is much more reliable than a sampled data of 100k with 500 $\theta$.
# 
