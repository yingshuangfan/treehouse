#!/usr/bin/env python
# coding: utf-8

# # Sampling
# 
# ## Direct Sampling
# 
# ## Likelihood(weighted) Sampling

# ## Rejection Sampling
# 
# Intuitive: A method to draw data samples from the target distribution $P$(e.g. posterior), by suggesting a easy-to-simulate distribution $Q$, such that generating data samples from $Q$ is much easier.
# 
# **Def. Rejection Sampling:** Given target distribution $P$ and candidate distribution $Q$, if we could find a constant $C$ such that it upper bounds the importance ratio $\frac{p(x)}{q(x)}$, such that:
# \begin{align*}
# C = \sup_x \frac{p(x)}{q(x)} \lt \infty
# \end{align*}
# 
# Then procedure of Rejection Sampling is defined as follows:
# 1. Simulate $x \sim Q$.
# 2. Simulate $u \sim Unif(0, 1)$, if $u \le \frac{p(x)}{Cq(x)} $ then accept it, otherwise discard it.
# 3. Repeat steps 1&2 until sufficient data samples are obtained.
# 
# Comments:
#  - The computation efficiency is highly related to the choice of candidate distribution $Q$. If the $Q$ is very different from the target distribution $P$, then Rejection Sampling method is not recommended, since majorith of the data samples would be discarded. 
#  - The Acceptance Rate:
# \begin{align*}
# P(accept) &= P\left(u \le \frac{p(x)}{Cq(x)}\right) \\
# &= E_u\left[I\left(u \le \frac{p(x)}{Cq(x)}\right)\right] \\
# &= E_q\left[E_u\left[I\left(u \le \frac{p(x)}{Cq(x)}\right)\mid x \right]\right] \\
# &= E_q\left[P\left(u \le \frac{p(x)}{Cq(x)}\mid x \right)\right] \\
# &= \int \frac{p(x)}{Cq(x)} q(x) \,dx \\
# &= \frac{1}{C}
# \end{align*}
#  - In common practice, the procedure to find candidate distribution and calcuate the constant $C$ is complicated, not to mention the accepance rate is extremely low for bad estimation. Thus Gibbs Sampler is recommended in general.
# 

# ## Importance Sampling
# 
# Intuitive: A method to estimate the expectation of a target function $f(X)$(e.g. posterior pdf), where random variable $X$ follows distribution $P$. Generally speaking, if we could draw enough data samples of $X$ from $P$, we could estimate $E_p[f]$ with the average of $f(z)$. However, **when it's rather difficult to simulate such data samples(the common case), Importance Sampling method provides a general methodology to estimate the expectation**. Notice that even though the name is related to sampling, but the goal here is to calculate the integration of $\int f(x)p(x) \,dx$.
# 
# **Def. Importance Sampling:** Find an easy-to-simulate candidate distribution $Q$ instead of $P$ to generate sufficient data samples from, and the expectation of the target function $f(X)$ could be estimated as:
# 
# \begin{align*}
# E_p[f] &= \int f(x)p(x) \,dx \\
# &= \int f(x)\frac{p(x)}{q(x)}q(x) \,dx \\
# &= \frac{1}{N} \sum_{i=1}^N f(z_i) \frac{p(z_i)}{q(z_i)} \\
# &z_i \sim Q \\
# \end{align*}
# 
# Comments:
#  - $w_i = \frac{p(z_i)}{q(z_i)}$ is also called the Importance Weights.
#  - Compared to Rejection Sampling, we notice that all data samples $z_i$ are accepted no matter how. In fact, if $z_i$ is located where the candidate distribution $Q$ diverges siginificantly from the original $P$, the sample value will be punished with a small weight instead.
#  - In common practice, the candidate distribution $Q$ should have lower variance than original distribution $P$. ?? Is it necessary for Q to be an envelope of P?

# ## Markov Chain Monte Carlo(MCMC)
# 
# **Def. Metropolis-Hasting Algorithm:** Given target function $f(X)$(e.g. posterior), candidate function $q(x, y)$(e.g. joint posterior) or $q(y|x)$(e.g. full conditional), the procedure of drawing data samples is:
# 1. Iinitialize $x^{(0)}$.
# 2. At the k-th iteration, simulate $y \sim q(x^{(k-1)}, y)$.
# 3. Calcuate the current acceptance rate:
# \begin{align*}
# \alpha(x^{(k-1)}, y) = min\left\{ \frac{f(y)q(y, x^{(k-1)})}{f(x^{(k-1)}) q(x^{(k-1)}, y)}, 1 \right\}
# \end{align*}
# 4. Simulate $u \sim Unif(0, 1)$. Update $x$ as follows:
# 
# \begin{align*}
# x^{(k)} = \left\{
# \begin{array}{ll}
#       y & u \le \alpha(x^{(k-1)}, y) \\
#       x^{(k-1)} & otherwise \\
# \end{array} 
# \right. 
# \end{align*}
# 
# **Corollary. Random Walk Metropolis:** The special case of Metropolis-Hasting Algorithm where $q(x, y) = q(y, x)$. 
#  - The probability of moving from x to y and y to x are identical. Thus we have the property of symmetric, and the acceptence rate could be simplified to:
#  \begin{align*}
# \alpha(x^{(k-1)}, y) = min\left\{ \frac{f(y)}{f(x^{(k-1)})}, 1 \right\}
# \end{align*}
# 
# **Corollary. Independent Metropolis:** The special case of Metropolis-Hasting Algorithm where $q(x, y) = p(y)$.
#  - The probability of moving from x to y is indepentent to x, in other words the probability of transfering to a certain state is independent to its previous states. The acceptence rate could be simplified to:
# \begin{align*}
# \alpha(x^{(k-1)}, y) = min\left\{ \frac{f(y)p(x^{(k-1)})}{f(x^{(k-1)})p(y)}, 1 \right\}
# \end{align*}
# 
# ### Gibbs Sampler
# 
# **Def. Gibbs Sampler:** Draw data samples from the joint distribution $p(\theta_1,\theta_2,...,\theta_m)$ by sampling $\theta_i$ iteratively from the full conditional distribution $p(\theta_i|\theta_{-i}) = p(\theta_i|\theta_1,\theta_2,...,\theta_{i-1},\theta_{i+1},...,\theta_m)$, with the procedure as follows:
# 1. Initialize $\theta_1^{(0)},\theta_2^{(0)},...,\theta_m^{(0)}$.
# 2. At the k-th iteration, for $i = 1,2,...m$, sample $\theta_i$ from:
# \begin{align*}
# \theta_i^{(k)} \sim p(\theta_i|\theta_1^{(k)},\theta_2^{(k)},...,\theta_{i-1}^{(k)},\theta_{i+1}^{(k-1)},...,\theta_m^{(k-1)})
# \end{align*}
# 
# Comments:
#  - Gibbs sampler is a special case of Metropolis-Hasting Algorithm where the acceptence rate is always 1. Proof: let $x=(\theta_i,\theta_{-i})$ and $y=(\theta_i^*,\theta_{-i})$, the transition probability $q(x,y)=p(\theta_i^*|\theta_{-i})$(e.g. full conditional), thus the acceptence rate can be derived as:
# \begin{align*}
# \alpha(x, y) &= min\left\{ \frac{f(y)q(y, x)}{f(x) q(x, y)}, 1 \right\} \\
# &= min\left\{ \frac{p(\theta_i^*,\theta_{-i})p(\theta_i|\theta_{-i})}{p(\theta_i,\theta_{-i}) p(\theta_i^*|\theta_{-i})}, 1 \right\} \\
# &= min\left\{ \frac{p(\theta_i^*,\theta_{-i})p(\theta_i,\theta_{-i})p(\theta_{-i})}{p(\theta_i,\theta_{-i}) p(\theta_i^*,\theta_{-i})p(\theta_{-i})}, 1 \right\} \\
# &= 1
# \end{align*} 
# 

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
# 3. Repeat steps 1&2 in order to generate the i.i.d sampled data $z_1,z_2,...,z_N$, giving the size of such dataset(N) is sufficiently large, then the average of sampled data is an Monte Carlo Estimation of new data point $x_{new}$.
# 
# Comments: 
#  - Notice that we only sample exactly ONE $z_i$ from any given $\theta_i$ sampled from the posterior. The point here is that the size of $z_i$ doesn't matter, as long as we have sufficent samples of $\theta_i$. In other words, the estimation here relies on taking the "average" over $\,d\theta$. For instance, a sampled data of 10k with 5k $\theta$ is much more reliable than a sampled data of 100k with 500 $\theta$.
# 
