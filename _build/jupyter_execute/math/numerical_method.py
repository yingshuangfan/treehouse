#!/usr/bin/env python
# coding: utf-8

# # Numerical Method
# 
# Goals:
#  - Design computational algorithms.
#  - Understand why they are efficient.
#  - Discuss their limitations.
#  - Design fast algorithms.
#  
# The limitations for DNN:
#  - Expressibility: how to explain a DNN succeeding to approximate a certain function?
#  - Learning and Generalization: how to explain why some optimization algorithms are better at generalization?
#  - Robustness and Stability: why a DNN could be sensitive to input features or training data?
# 

# ## Convolutional Neural Network
# 
# ### Convolution Operators
# 
# **Def.1-1 1-d Convolution for continuous functions:** Given two functions $x(u)$ and $h(u)$, their convolution is defined as:
# 
# \begin{align*}
# c(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(u)h(t-u) \,du
# \end{align*}
# 
# **Def.1-2 1-d Convolution for discrete sequences:** Given two sequences $a = (a_0,a_1,...,a_{m-1})$ and $b = (b_0,b_1,...,b_{m-1})$, their convolution is defined as:
# 
# \begin{align*}
# c_{k} = \sum_{i=0}^k a_i b_{k-i}
# \end{align*}
# 
# **Corollary.1-2-1 Matrix Form of 1-d Convolution for discrete sequences:** If $c = (c_0,c_1,...c_{m-1})$, where $c_k$ is defined as in Def.1-2, then the convolution operation could be written as matrix form:
# 
# \begin{align*}
# c = Ka = 
# \begin{bmatrix}
# a_0 b_0\\
# a_0 b_1 + a_1 b_0\\
# ...\\
# a_0 b_{m-1} + a_1 b_{m-2} + ... + a_{m-1}b_0
# \end{bmatrix} =
# \begin{bmatrix}
# b_0\\
# b_1 & b_0\\
# ...\\
# b_{m-1} & b_{m-2} & ... & b_0
# \end{bmatrix} a
# \end{align*}
# 
# Comments:
#  - The first function $x(u)$ or sequence $a$ is referred as the "input", while the second function $h(u)$ or sequence $b$ is referred as the "kernel".
#  - The definition of convolution could be further generalize to higher dimension.
#  
# 
#  
#  
# **Def.1-3 2-d Convolution for continuous functions:** SKIPPED
# 
# **Def.1-4 2-d Convolution for matrices:** Given two (finite) matrices $I \in \mathcal{R}^{m_1\times n_1}$ and $K \in \mathcal{R}^{m_2\times n_2}$, their convolution $S \in \mathcal{R}^{(m_1+m_2-1)\times(n_1+n_2-1)}$ is defined as:
# 
# \begin{align*}
# S(i,j) = (I*K)(i,j) =  \sum_{k_1=0}^{m_1-1} \sum_{k_2=0}^{n_1-1} I(k_1,k_2)K(i-k_1,j-k_2)
# \end{align*}
# 
# Reference: WIKI: "Multidimensional convolution with one-dimensional convolution methods"
# 
# 
# **Corollary.1-4-1 Matrix Form of 2-d Convolution for matrices:** For $S \in \mathcal{R}^{(m_1+m_2-1)\times(n_1+n_2-1)}$ as defined in Def.1-4, the convolution operation could be written as matrix form, where $T \in \mathcal{R}^{(m_1+m_2-1)\times(n_1+n_2-1)}$:
# 
# 1. Let $T_k \in \mathcal{R}^{(n_1+n_2-1)\times n_1}$ be the k-th (sub)block of matrix $T$, where $k=1,2,...,n_1$. $T_k$ is a Toeplitz matrix constructed by the k-th row of matrix $H$:
# 
# \begin{align*}
# H = 
# \begin{bmatrix}
# K & 0\\
# 0 & 0
# \end{bmatrix}
# \end{align*}
# 
# For instance, the k-th block takes the form, where $K_{ij}$ denotes the element of $K[i][j]$:
# \begin{align*}
# T_k =
# \begin{bmatrix}
# K_{k0} & 0 & ... \\
# K_{k1} & K_{k0} & 0 & ... \\
# K_{k2} & K_{k1} & K_{k0} & 0 & ... \\
# \vdots & \ddots & \ddots \\
# K_{km_1} & K_{k(m_1-1)} & ... \\
# 0 & K_{km_1} & K_{k(m_1-1)} & ...\\
# \vdots & \ddots & \ddots \\
# \end{bmatrix}
# \end{align*}
# 
# 2. Generate the complete doubly-block Toeplitz matrix $T \in \mathcal{R}^{(m_1+m_2-1)(n_1+n_2-1)\times m_1 n_1}$
# 
# \begin{align*}
# T = 
# \begin{bmatrix}
# T_1 & T_2 & ... & T_{m1}\\
# T_2 & T_1 & ... & T_{m1-1}\\
# \vdots & \ddots \\
# T_{m1+m2-1} & ... 
# \end{bmatrix}
# \end{align*}
# 
# 3. Let vector $I^* \in \mathcal{R}^{m_1 n_1}$ denotes the flatten input matrix $I$, and vector $S^* \in \mathcal{R}^{(m_1+m_2-1)(n_1+n_2-1)}$ denotes the flatten convolutional result matrix $S$, then the convolution operation can be presented by matrix dot operation:
# \begin{align*}
# S^* = T I^*
# \end{align*}
# 
# 
# Comments:
#  - The definition we mentioned in Def.1-2 and Def.1-4 are element-wise. However, for those indices which are invalid in the original matrix or sequence, we could assume the values are zeros.
#  - Even though the result of convolution has much higher dimension by definition, it is usually very sparse and could be represented by low dimensional vector(s) instead. 
# 
# 

# ### Kernel Methods
# 
# **Def.2-1 Positive Definitive(PD) Kernels:** A positive kernel defined on set $\mathcal{X}$ should satisfied: 
#  - symmetric: $K(x,x') = K(x,x')$.
#  - positive definite: for any co-efficients $(a_1,a_2,...,a_N) \in \mathcal{R}^N$, 
#     \begin{align*}
#     \sum_{i=1}^N \sum_{j=1}^N a_i a_j K(x^{(i)},x^{(j)}) \ge 0
#     \end{align*}
#     
# Example:
#  - $\mathcal{X} = \mathcal{R}$, $K(x,x')=xx'$. Proof:
#     \begin{align*}
#     \sum_{i=1}^N \sum_{j=1}^N a_i a_j x^{(i)}x^{(j)} = \sum_{i=1}^N a_i x^{(i)} (\sum_{j=1}^N  a_j x^{(j)})= (\sum_{i=1}^N a_i x^{(i)})^2 \ge 0
#     \end{align*}
#  - $\mathcal{X} = \mathcal{R}^d$, $K(x,x')=\langle x,x'\rangle $. Proof:
#     \begin{align*}
#     \sum_{i=1}^N \sum_{j=1}^N a_i a_j \langle x^{(i)},x^{(j)} \rangle = \sum_{i=1}^N \sum_{j=1}^N a_i a_j \sum_{k=1}^d x_k^{(i)}x_k^{(j)} = \sum_{k=1}^d \sum_{i=1}^N a_i x_k^{(i)}(\sum_{j=1}^N a_j x_k^{(j)}) = \sum_{k=1}^d (\sum_{i=1}^N a_i x_k^{(i)})^2 \ge 0
#     \end{align*}
#  - $\mathcal{X}$ ia ANY set, $\phi: \mathcal{X}^2 \to \mathcal{R}^d$, $K(x,x')=\langle \phi(x),\phi(x')\rangle$. Proof:
#     \begin{align*}
#     \sum_{i=1}^N \sum_{j=1}^N a_i a_j \langle \phi(x^{(i)}),\phi(x^{(j)}) \rangle = \sum_{i=1}^N \sum_{j=1}^N a_i a_j \sum_{k=1}^d \phi(x^{(i)})_k \phi(x^{(j)})_k = \sum_{k=1}^d \sum_{i=1}^N a_i \phi(x^{(i)})_k(\sum_{j=1}^N a_j \phi(x^{(j)})_k )= \sum_{k=1}^d (\sum_{i=1}^N a_i \phi(x^{(i)})_k)^2 \ge 0
#     \end{align*}
