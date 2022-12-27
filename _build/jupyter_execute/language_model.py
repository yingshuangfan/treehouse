#!/usr/bin/env python
# coding: utf-8

# ### Probabilistic Language Model
# 
# **Def.1** Probability of sentence: $W=(w_1,...,w_n)$ as a sequence of words, by chain-rule we have 
# 
# \begin{align}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_0,...,w_{i-1}).
# \end{align}
# 
#  - Where is $w_0$? by definition, we ignore $P(w_0)$ in the joint probability. 
#  - How to estimate conditional probability $P(w_i|w_0,...,w_{i-1})$? Maximum likelihood estimate, where $cnt(w_0,...,w_i)$ is defined as the number of counts for sequence $w_0,...,w_i$ in the corpus. 
#     \begin{align}
#     P(w_i|w_0,...,w_{i-1}) = \frac{cnt(w_0,...,w_i)}{cnt(w_0,...,w_{i-1})}.
#     \end{align}
#  - To obtain a good estimation, the corpus needs to be sufficiently large, considering the great number of possible sequnces of words. (which is often NOT realistic!)

# #### Maximum Likelihood Estimate(MLE)
# 
# Intuition: The method define the way to determine the parameters of a model, such that the likelihood of the process described by the model is maximized based on the data that we have oberserved.

# #### N-gram Model
# 
# Intuition: decrease the number of possible sequences of words, by adoption the Markov Assumption.
#     
# **Assumption.1 Markov Assumption:** the future word only depends on the previous K words.
# \begin{align}
# P(w_i|w_0,...,w_{i-1}) = P(w_i|w_{i-K},...,w_{i-1}).
# \end{align}
# 
# **Def.2 N-gram model** Given the assumption.1, now the probability of a sentence W in def.1 can be simplified as:
# \begin{align}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_{i-K},...,w_{i-1}).
# \end{align}
# \begin{align}
# P(w_i|w_0,...,w_{i-1}) = \frac{cnt(w_{i-K},...,w_i)}{cnt(w_{i-K},...,w_{i-1})}.
# \end{align}
# 
# **Corollary.1 Unigram model** Let K=0, thus each word is independent(where total is the total number of sequences in the corpus). e.g. Bag-of-Words model.
# \begin{align}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i) = \prod_{i=1}^n \frac{cnt(w_i)}{total}.
# \end{align}
# 
# **Corollary.2 Bigram model** Let K=1, thus each word is only dependent to the previous word.
# \begin{align}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_{i-1}) = \prod_{i=1}^n \frac{cnt(w_i,w_{i-1})}{cnt(w_{i-1})}.
# \end{align}

# #### Evaluation Metric
# 
# The fitted language model M is **a set of conditional probabilities**!
# This section we discuss few metrics to evaluate the performance of the fitted language model that could be used on a test dataset.
# 
# **Def.3 Perplexity:** Defined the sentence probability normalized by the number of words $n$.
# \begin{align}
# PP(W) = \sqrt[n]{P(w_1,...,w_n)} = \sqrt[n]{\prod_{i=1}^n P(w_i|w_{i-K},...,w_{i-1})}.
# \end{align}
# 
#  - Perplexity is closely related to the sentence probability. In fact, **maximize the sentence probability equals to minimize perplexity**. When comparing different models on a given test dataset, the smaller perplexity yields the better model.
#  - We often use the log of perplexity to overcome the overflow problem:
#  
# \begin{align}
# \log{PP(W)} = -\frac{1}{n} \sum_{i=1}^n \log{P(w_i|w_{i-K},...,w_{i-1})}.
# \end{align}
# 
# **Def.4 Entropy:** The entropy of a random variable X with the probability distribution p is defined as:
# 
# \begin{align}
# H(p) = E_X[-\log_{2}{p(X)}] = -\sum_{x \in X}p(x)\log_{2}{p(x)}.
# \end{align}
# 
#  - Entropy describe **the average level of information(uncertainty)** given the possible outcomes of a random variable. For example, the maximum entropy is obtained when X follows a uniformed distribution, while the minimum is obtained when X equals to a fixed value(pdf=single point mass).
#  - For a valid distribution p, the entropy would always be non-negative! 
# 
# **Assumption.2 Asymptotic Equipartition Property :** Given a discrete-time ergodic stochastic process X:
# \begin{align}
# \lim_{n \to \infty}-\frac{1}{n}\log_{2}{P(X_1,X_2,...X_n)} \to H(X)
# \end{align}
# 
#  - The property can be proved by Shannon-McMillan-Breiman Theorem. [wiki](https://en.wikipedia.org/wiki/Asymptotic_equipartition_property#Discrete-time_finite-valued_stationary_ergodic_sources)
#  - It states that although there are many possible results that could be produced by the random process, the one we actually observed is most probable from a set of outcomes where each one has the approximately same probability. Thus, the assumption proposes that the large deviation from mean(if exists) would decay exponentially with the increasing number of data samples.
# 
# 
# **Corollary.3 Entropy for language model:** The probability distribution p is defined as the probability language model M. Therefore, the entropy of a sequence of words is defined as:
# 
# \begin{align}
# H(M) = \lim_{n \to \infty}{H_n(W_{1:n})} = \lim_{n \to \infty}-\frac{1}{n}\sum_{W_{1:n}}P(W_{1:n})\log_{2}{P(W_{1:n})}.
# \end{align}
# 
# Given the assumption.2, it could be simplified as:
# 
# \begin{align}
# H(M) = \lim_{n \to \infty}-\frac{1}{n}\log_{2}{P(W_{1:n})}.
# \end{align}
# 
# **Def.5 Cross-Entropy:** The cross-entropy between two distributions p and q over the same random variable X is defined as:
# 
# \begin{align}
# H(p,q) = E_{p(X)}[-\log_{2}{q(X)}] = -\sum_{x \in X}p(x)\log_{2}{q(x)}.
# \end{align}
# 
#  - $H(p,q) \ge H(p,p)=H(p)$
#  - It could measure the divergence between two distribution.
#  
#  
# **Corollary.4 Cross-Entropy for language model:** Suppose M is the fitted language model from the training dataset, and L is the real language model that we pursue. The goal is to minimize the cross-entropy between M and L. Denote S as the sequence in corpus, the cross-entropy is defined as:
# 
# \begin{align}
# H(L,M) = E_{L(S)}[-\log_{2}{M(S)}] = -\lim_{n \to \infty}\log_{2}{M(W_{1:n})}.
# \end{align}
# 
#  - $\log_{2}{PP(M)}=H(L,M).$ The perplexity of a fitted language model could be computed with its cross-entropy(from the real language model).
#  - In a finite dataset T with N samples, the estimate of cross-entropy can be computed with:
#     \begin{align}
#     H(T,M) = E_{T(S)}[-\log_{2}{M(S)}] = -\frac{1}{N}\sum_{i=1}^N \frac{1}{|S_i|}\log_{2}{M(S_i)}.
#     \end{align}

# #### Smoothing
# 
# Intuitive: the problem of zeros. It is very common that the sentence in a test dataset does NOT exist in the training dataset, however the N-gram language model would output zero probability! By definition, the zero in probability could result in failure when computing the perplexity or entropy for a given language model. Therefore, we introduce smoothing to eliminate zeros in the model.
# 
# 
# **Def.6 Laplace Smoothing:** Add a fixed number $\lambda$ when computing the conditional probability, where V is the size of vocabulary(unique words) for the corpus. The revised estimation of conditional probability is:
# 
# \begin{align}
# P(w_i|w_{i-K},...,w_{i-1}) = \frac{cnt(w_{i-K},...,w_i)+\lambda}{cnt(w_{i-K},...,w_{i-1})+\lambda V}.
# \end{align}
# 
#  - $\lambda$ can be float! e.g. $\lambda=\frac{1}{V}$
#  - Add-one estimation(continuity correction): Let $\lambda=1$. However this approach is not recommended if V is too large.
