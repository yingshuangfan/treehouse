#!/usr/bin/env python
# coding: utf-8

# # Probabilistic Language Model
# 
# ## Probability of Sentence
# 
# **Def.1** Probability of sentence: $W=(w_1,...,w_n)$ as a sequence of words, by chain-rule we have 
# 
# \begin{align*}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_0,...,w_{i-1})
# \end{align*}
# 
#  - Where is $w_0$? by definition, we ignore $P(w_0)$ in the joint probability. 
#  - How to estimate conditional probability $P(w_i|w_0,...,w_{i-1})$? Maximum likelihood estimate, where $cnt_{seq}$ is defined as the number of counts for sequence $w_0,...,w_i$ in the corpus. 
#     \begin{align*}
#     P(w_i|w_0,...,w_{i-1}) = \frac{cnt_{seq}(w_0,...,w_i)}{cnt_{seq}(w_0,...,w_{i-1})}
#     \end{align*}
#  - To obtain a good estimation, the corpus needs to be sufficiently large, considering the great number of possible sequnces of words. (which is often NOT realistic!)

# ## Maximum Likelihood Estimate(MLE)
# 
# Intuition: The method define the way to determine the parameters of a model, such that the likelihood of the process described by the model is maximized based on the data that we have oberserved.

# ## N-gram Model
# 
# Intuition: decrease the number of possible sequences of words, by adoption the Markov Assumption.
#     
# **Assumption.1 Markov Assumption:** the future word only depends on the previous K words.
# \begin{align*}
# P(w_i|w_0,...,w_{i-1}) = P(w_i|w_{i-K},...,w_{i-1})
# \end{align*}
# 
# **Def.2 N-gram model** Given the assumption.1, now the probability of a sentence W in def.1 can be simplified as:
# \begin{align*}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_{i-K},...,w_{i-1})
# \end{align*}
# \begin{align*}
# P(w_i|w_0,...,w_{i-1}) = \frac{cnt_{seq}(w_{i-K},...,w_i)}{cnt_{seq}(w_{i-K},...,w_{i-1})}
# \end{align*}
# 
# **Corollary.2-1 Unigram model** Let K=0, thus each word is independent(where $total_{seq}$ is the total number of sequences in the corpus). e.g. Bag-of-Words model.
# \begin{align*}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i) = \prod_{i=1}^n \frac{cnt_{seq}(w_i)}{total_{seq}}
# \end{align*}
# 
# **Corollary.2-2 Bigram model** Let K=1, thus each word is only dependent to the previous word.
# \begin{align*}
# P(W) = P(w_1,...,w_n) = \prod_{i=1}^n P(w_i|w_{i-1}) = \prod_{i=1}^n \frac{cnt_{seq}(w_i,w_{i-1})}{cnt_{seq}(w_{i-1})}
# \end{align*}

# ## Text Classification
# 
# **Def.3 Bag-of-words model:** In this model, any text is represented as a set of its (unique)words, ignoring its grammar or sequence ordering. Represent document $D \to \{w_i, cnt_i\}_{i=1}^V$, where $w_i$ is the i-th unique word, $cnt_i$ is the number of occurence of $w_i$, $V$ is the total size of vocabulary.
#  - By this method, we can easily represent text documents into vectors.
#  - It is widely used in general text classification, where the occurency of words is the key feature.
#  
# **Def.4 Text Classification:** Given labeled pairs of docuemnt $d_i$ and its class label $c_i$ as the training data, learn a model which output a predicted class $c_p$ for any input document $d_p$.
#  - The MAP estimator of Def.7, can be defined as:
#     \begin{align*}
#     \DeclareMathOperator*{\argmax}{argmax}
#     c_{MAP} = \argmax_{c}{P(c|d)} = \argmax_{c}{P(d|c)P(c)}
#     \end{align*}
#    Notice that by bayes rule, the marginal probability $P(d)$ is ignored as constant(regarding class c).

# ## Maximum a Posterior Estimation(MAP)
# 
# Intuition: In bayesian statistics, MAP method is an estimate that equals the mode of the posterior distribution. Compared to MLE, MAP method introduces the prior distribution into the estimation which represents our former knowledge of the data samples. Therefore, MAP can be viewed as a regularization of MLE.
# 

# ## Naive Bayes
# 
# **Def.5 Naive Bayes Estimator:** Based on Bag-of-words assumption in Def.3, the MAP estimator is defined as:
# \begin{align*}
# \DeclareMathOperator*{\argmax}{argmax}
# c_{NB} = \argmax_{c}{P(d|c)P(c)} = \argmax_{c}{P(c)\prod_{i}{P(w_i|c)}}
# \end{align*}
#     
# - To prevent the underflow problem, we often use the log of probability. Thus the estimator can be rewritten as:
#     \begin{align*}
#     \DeclareMathOperator*{\argmax}{argmax}
#     c_{NB} = \argmax_{c}\log_{2}{P(d|c)P(c)} = \argmax_{c}{\{\log_{2}{P(c)} + \sum_{i}{\log_{2}P(w_i|c)}\}}
#     \end{align*}
#     
# - To estimate the $P(c)$ and $P(w_i|c)$, we adopt the MLE method once again: where $cnt_{doc}$ denotes the count of documents, $cnt_{word}$ denotes the count of words, and $total_{doc}$ denotes the total number of documents(size of corpus).
#     \begin{align*}
#     P(c) = \frac{cnt_{doc}(c_i=c)}{total_{doc}} \\
#     P(w_i|c) = \frac{cnt_{word}(w_i, c_i=c)}{cnt_{word}(c_i=c)}
#     \end{align*}
#     

# ## Evaluation Metric
# 
# The fitted language model M is **a set of conditional probabilities**!
# This section we discuss few metrics to evaluate the performance of the fitted language model that could be used on a test dataset.
# 
# **Def.6 Perplexity:** Defined the sentence probability normalized by the number of words $n$.
# \begin{align*}
# PP(W) = \sqrt[n]{P(w_1,...,w_n)} = \sqrt[n]{\prod_{i=1}^n P(w_i|w_{i-K},...,w_{i-1})}
# \end{align*}
# 
#  - Perplexity is closely related to the sentence probability. In fact, **maximize the sentence probability equals to minimize perplexity**. When comparing different models on a given test dataset, the smaller perplexity yields the better model.
#  - We often use the log of perplexity to overcome the overflow problem:
#  
# \begin{align*}
# \log{PP(W)} = -\frac{1}{n} \sum_{i=1}^n \log{P(w_i|w_{i-K},...,w_{i-1})}
# \end{align*}
# 
# **Def.7 Entropy:** The entropy of a random variable X with the probability distribution p is defined as:
# 
# \begin{align*}
# H(p) = E_X[-\log_{2}{p(X)}] = -\sum_{x \in X}p(x)\log_{2}{p(x)}.
# \end{align*}
# 
#  - Entropy describe **the average level of information(uncertainty)** given the possible outcomes of a random variable. For example, the maximum entropy is obtained when X follows a uniformed distribution, while the minimum is obtained when X equals to a fixed value(pdf=single point mass).
#  - For a valid distribution p, the entropy would always be non-negative! 
# 
# **Assumption.2 Asymptotic Equipartition Property :** Given a discrete-time ergodic stochastic process X:
# \begin{align*}
# \lim_{n \to \infty}-\frac{1}{n}\log_{2}{P(X_1,X_2,...X_n)} \to H(X)
# \end{align*}
# 
#  - The property can be proved by Shannon-McMillan-Breiman Theorem. [wiki](https://en.wikipedia.org/wiki/Asymptotic_equipartition_property#Discrete-time_finite-valued_stationary_ergodic_sources)
#  - It states that although there are many possible results that could be produced by the random process, the one we actually observed is most probable from a set of outcomes where each one has the approximately same probability. Thus, the assumption proposes that the large deviation from mean(if exists) would decay exponentially with the increasing number of data samples.
# 
# 
# **Corollary.7-1 Entropy for language model:** The probability distribution p is defined as the probability language model M. Therefore, the entropy of a sequence of words is defined as:
# 
# \begin{align*}
# H(M) = \lim_{n \to \infty}{H_n(W_{1:n})} = \lim_{n \to \infty}-\frac{1}{n}\sum_{W_{1:n}}P(W_{1:n})\log_{2}{P(W_{1:n})}
# \end{align*}
# 
# Given the assumption.2, it could be simplified as:
# 
# \begin{align*}
# H(M) = \lim_{n \to \infty}-\frac{1}{n}\log_{2}{P(W_{1:n})}
# \end{align*}
# 
# **Def.8 Cross-Entropy:** The cross-entropy between two distributions p and q over the same random variable X is defined as:
# 
# \begin{align*}
# H(p,q) = E_{p(X)}[-\log_{2}{q(X)}] = -\sum_{x \in X}p(x)\log_{2}{q(x)}
# \end{align*}
# 
#  - $H(p,q) \ge H(p,p)=H(p)$
#  - It could measure the divergence between two distribution.
#  
#  
# **Corollary.8-1 Cross-Entropy for language model:** Suppose M is the fitted language model from the training dataset, and L is the real language model that we pursue. The goal is to minimize the cross-entropy between M and L. Denote S as the sequence in corpus, the cross-entropy is defined as:
# 
# \begin{align*}
# H(L,M) = E_{L(S)}[-\log_{2}{M(S)}] = -\lim_{n \to \infty}\log_{2}{M(W_{1:n})}
# \end{align*}
# 
#  - $\log_{2}{PP(M)}=H(L,M).$ The perplexity of a fitted language model could be computed with its cross-entropy(from the real language model).
#  - In a finite dataset T with N samples, the estimate of cross-entropy can be computed with:
#     \begin{align*}
#     H(T,M) = E_{T(S)}[-\log_{2}{M(S)}] = -\frac{1}{N}\sum_{i=1}^N \frac{1}{|S_i|}\log_{2}{M(S_i)}
#     \end{align*}

# ## Smoothing
# 
# Intuitive: the problem of zeros. It is very common that the sentence in a test dataset does NOT exist in the training dataset, however the N-gram language model would output zero probability! By definition, the zero in probability could result in failure when computing the perplexity or entropy for a given language model. Therefore, we introduce smoothing to eliminate zeros in the model.
# 
# 
# **Def.9 Laplace Smoothing:** Add a fixed number $\lambda$ when computing the conditional probability, where V is the size of vocabulary(unique words) for the corpus.
# 
#  - $\lambda$ can be float! e.g. $\lambda=\frac{1}{V}$
#  - Add-one estimation(continuity correction): Let $\lambda=1$. However this approach is not recommended if V is too large.
# 
# **Corollary.8-1 Laplace Smoothing for N-gram model:**
# \begin{align*}
# P(w_i|w_{i-K},...,w_{i-1}) = \frac{cnt_{seq}(w_{i-K},...,w_i)+\lambda}{cnt_{seq}(w_{i-K},...,w_{i-1})+\lambda V}
# \end{align*}
# 
# **Corollary.8-2 Laplace Smoothing for Naive Bayes model:**
# \begin{align*}
# P(w_i|c_i=c) = \frac{cnt_{word}(w_i, c_i=c)+\lambda}{cnt_{word}(c_i=c)+\lambda V}
# \end{align*}
# 
