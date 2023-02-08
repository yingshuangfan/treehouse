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
# Comments:
#  - The length of two sequences can be different, we could assume that "m" in the definition is the maximum length of two sequences, where the rest elements are zeros.
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
# **Def.1-5 2-d Convolution for matrices in Cross-Correlation Form:** The convolution operation defined in 1-4 where we flipped the kernel w.r.t the input is useful for writing proofs. On the other hand, the so-called "Cross-Correlation" version of operation is most implemented for DNN in practice. The main reason is that the size of kernal is much smaller compared to real-world dataset, thus the output size is also more compact by the later definition.
# 
# \begin{align*}
# S(i,j) = (I*K)(i,j) =  \sum_{k_1=0}^{m_2-1} \sum_{k_2=0}^{n_2-1} I(i+k_1,j+k_2)K(k_1,k_2)
# \end{align*}
# 
# Comments: Notice that the size of output $S \in \mathcal{R}^{(m_1-m_2+1)\times(n_1-n_2+1)}$ is smaller than the definition in Def.1-4. However, if we assume that the size of input is close to infinite(big dataset), then the difference could be ignored.
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

# ## Kernel Methods
# 
# Intuition: Develop a vasetile algorithm without making any assumptions on the training dataset. (Recall the intuition for non-parametric bayesian)
# 
# **Def.2-1 Positive Definitive(p.d.) Kernels:** A positive kernel defined on set $\mathcal{X}$ should satisfied: 
#  - symmetric: $K(x,x') = K(x,x')$.
#  - positive definite: for any co-efficients $(a_1,a_2,...,a_N) \in \mathcal{R}^N$, 
#     \begin{align*}
#     \sum_{i=1}^N \sum_{j=1}^N a_i a_j K(x^{(i)},x^{(j)}) \ge 0
#     \end{align*}
#     
# **Corollary.2-1-1 Proof of p.d. Kernels by PSD matrix:** Let the **similarity matrix K** be defined as $[K]_{ij} = K(x^i,x^j)$, then the kernel is positive definite if and only if its similarity matrix is **positive semi-definite**.
#  - Similarity matrix K is symmetric by definition, thus we could apply eigen-decomposition on K.
#     
# Example: since symmetry is very easy to observe, we only illustrate the proof of non-negativity.
#  - $\mathcal{X} = \mathcal{R}$, $K(x,x')=xx'$. Proof:
#     \begin{align*}
#     &\sum_{i=1}^N \sum_{j=1}^N a_i a_j x^i x^j \\
#     &= \sum_{i=1}^N a_i x^i (\sum_{j=1}^N  a_j x^j) \\
#     &= (\sum_{i=1}^N a_i x^i)^2 \ge 0
#     \end{align*}
#  - $\mathcal{X} = \mathcal{R}^d$, $K(x,x')=\langle x,x'\rangle $. Proof:
#     \begin{align*}
#     &\sum_{i=1}^N \sum_{j=1}^N a_i a_j \langle x^i,x^j \rangle \\
#     &= \sum_{i=1}^N \sum_{j=1}^N \langle a_i x^i,a_j x^j \rangle \\
#     &= \langle \sum_{i=1}^N a_i x^i, \sum_{j=1}^N a_j x^j \rangle \\
#     &= \| \sum_{i=1}^N a_i x^i \|_d^2  \ge 0
#     \end{align*}
#  - $\mathcal{X}$ ia ANY set, $\phi: \mathcal{X}^2 \to \mathcal{R}^d$, $K(x,x')=\langle \phi(x),\phi(x')\rangle$. Proof:
#     \begin{align*}
#     &\sum_{i=1}^N \sum_{j=1}^N a_i a_j \langle \phi(x^{(i)}),\phi(x^{(j)}) \rangle \\
#     &= \sum_{i=1}^N \sum_{j=1}^N \langle a_i\phi(x^{(i)}),a_j\phi(x^{(j)}) \rangle \\
#     &= \langle \sum_{i=1}^N a_i\phi(x^{(i)}),\sum_{j=1}^N a_j\phi(x^{(j)}) \rangle \\
#     &= \| \sum_{i=1}^N a_i\phi(x^{(i)}) \|_d^2 \ge 0
#     \end{align*}
# 

# ### Aronszajna Theorem
# 
# Kernel K is PD if and only if we could find:
#  1. a Hilbert Space $\mathcal{H}$.
#  2. a mapping $\phi: \mathcal{X} \to \mathcal{H}$.
# 
# such that for any pair of $x,x' \in \mathcal{X}$, we have the kernel value(distance between x and x' as defined by kernel) equals to the inner product of mapping vectors:
#     \begin{align*}
#     K(x,x') = \langle \phi(x),\phi(x') \rangle_\mathcal{H}
#     \end{align*}
#     
# **Comments:** 
#  - The inner product is defined by the Hilbert Space we proposed. 
#  - The proof of theorem is complicated when the dimension of dataset $\mathcal{X}$ is infinite, for the finite case we could use simple linear algebra for proof. Thus we introduce the idea of RKHS to help us simplify the procedure.
#  - In fact, the RKHS and kernel are uniquely determined to each other. 
#  
# **Proof: (for the FINITE case)** Conversely proof TBC.
# 
# Suppose the dataset has N data points, thus the similarity matrix has dimension $N \times N$ and could be diagnalized on an orthonormal(orthogonal + normal) basis of eigenvector $(u_1,u_2,...u_N)$. For a p.d. kernel, we know its similarity matrix is p.s.d, thus the eigenvalues are non-genative with $\lambda_N \ge \lambda_{N-1} \cdots \ge \lambda_1 \ge 0$. Thus the $K[i,j]$ element could be written as:
# 
# \begin{align*}
# K(x_i,x_j) = K_{ij} = \left(\sum_{k=1}^N \lambda_k u_k u_k^T\right)_{ij} = \sum_{k=1}^N \lambda_k (u_k)_i (u_k)_j
# \end{align*}
# 
# Suppose we define a mapping $\phi$ such that $K(x_i,x_j) = \langle \phi(x_i),\phi(x_j) \rangle_{\mathcal{R}^N}$. By observation, let $\phi(x_i) = (\sqrt \lambda_1 (u_1)_i, \cdots, \sqrt \lambda_N (u_N)_i)$. In summary, in finite dimensional hilbert space, we could always find a function $\phi$ which statisfied the aforementioned statement given any p.d. kernel(and vice versa), thus the theorem is proved.
# 
# **Proof: (for the INFINITE case, using the concept of RKHS)** Conversely proof TBC.
# 
# If kernel function K is p.d. on dataset $\mathcal{X}$, then it is a reproducing kernel of a hilbert space $\mathcal{H} $. Let function $\phi: \mathcal{X} \to \mathcal{H}$ be the kernel map defined by $\phi(x) = K_x$ at any point x. Thus by the reproducing property, we have $f(x) = \langle f,K_x \rangle_\mathcal{H}$. In other words, for any function f fixed at another point y, we have $K(x,y) = \langle \phi(x),\phi(y) \rangle_\mathcal{H}$, thus the theorem is proved.
# 
# 
# 
# ### Repoducing Kernel Hilbert Space(RKHS)
#  
# **Def.2-2 Reproducing Kernel:** The function $K: \mathcal{X}^2 \to \mathcal{R}$ is called a repoducing kernel of hilbert space $\mathcal{H}$ if it satisfies:
#  1. If we fixed point $x$ for kernel function $K(x,x')$, then all functions $K_x: \mathcal{X} \to \mathcal{R}$ are contained in $\mathcal{H}$.
#  2. For every point $x$ and every function $f$ in $\mathcal{H}$, the **repoducing property** holds: 
#     \begin{align*}
#     f(x) = \langle f,K_x \rangle_\mathcal{H}
#     \end{align*}
#     
# Comments:
#  - RKHS is a special subset where the repoducing property holds. This property is crucial for the proof of Aronszajna Theorem.
#  - The kernel is called "reproducing", because it could repoduce the value of any function f at point x with the inner-product between f and kernel function fixed at point x. 
#  - The mapping function $\phi(x) = K_x$ maps the finite dataset $\mathcal{X}$ to a high-dimensional and infinite hilbert space $\mathcal{H}$. We call the hilbert space is infinite because the number of functions in the space is unlimited.
# 
# **Corollary.2-2-1 Uniqueness of repoducing kernel and RKHS:** There are **uniquely defined**! If $\mathcal{H}$ is a RKHS, then it has a unique repoducing kernel. Conversely, if a function K could be the repoducing kernel of at most one RKHS.
# 
# Proof: TBC.
# 
# **Corollary.2-2-2 p.d. Kernel = reproducing Kernel:** A function K is positive definite if and only if it is a reproducing kernel(of a RKHS).
# 
# Proof: TBC.
#  
# **Def.2-3 Alternative definition of RKHS:** A hilbert space $\mathcal{H}$ is a RKHS if and only if, for any point x in dataset $\mathcal{X}$, the linear mapping F as defined is continuous:
# \begin{align*}
# F: & \mathcal{H} \to \mathcal{R} \\
# & f \to f(x)
# \end{align*}
# 
# **Corollary.2-3-1 Pointwise Convergence of RKHS:** The convergence of RKHS is always pointwise convergence. If a series of function $\{f_n\}$ converges to function $f$, then we could assume that at any point x, we have series $\{f_n(x)\}$ converges to value $f(x)$.
# 
# **Proof:** TBC.
# 
# **Property:** Uniqueness of repoducing kernel and RKHS.
# 

# ### RKHS vs. Kernel
# 
# Intuition: If a certain kernel function is proposed, how to find the RKHS for such kernel? Supposed that the kernel is positive definite. The general procedure of finding a RKHS is:
# 
#  1. Look for an definition of inner-product.
#  2. Propose a candidate RKHS $\mathcal{H}$(with the inner-product).
#  3. Check if space $\mathcal{H}$ is a hilbert space.
#  4. Check if space $\mathcal{H}$ is a RKHS.
# 
# 
# #### Linear Kernel
# Task: find a RKHS for the defined kernel: $K(x,x')=\langle x,x' \rangle_{\mathcal{R}^d}$, where the dataset is defined in $\mathcal{R}^d$.
# 
# TBC.
# 
# 
# #### Polynomial Kernel (of degree 2)
# Task: find a RKHS for the defined kernel: $K(x,x')=\langle x,x' \rangle_{\mathcal{R}^d}^2$, where the dataset is defined in $\mathcal{R}^d$.
# 
# TBC.
# 
# #### polynomial Kernel (general case)
# Task: find a RKHS for the defined kernel: $K(x,x')=\langle x,x' \rangle_{\mathcal{R}^d}^n$, where the dataset is defined in $\mathcal{R}^d$.
# 
# TBC.

# ### Combining Kernels
# 
# Intuition: Some form of combination on kernels could still hold the property of positive definite. Thus we could take the advantage and skip the long proof from scratch.
# 
#  - $K_1 + K_2$
#  - $K_1 K_2$
#  - $c K_1$, where c is constant.
#  - K(x,x') = \lim_{n \to \infty} K_i(x,x'), where {K_i} is a sequence of p.d. kernels that converges pointwisely to a function K.
#  
#  
# ### Smoothness of the RKHS
# 
# **Def.2-4 Smoothness:** The norm of a function in the RKHS controlls fast the function varies over the dataset $\mathcal{X}$ with respect to the geometry defined by the kernel function.
# 
# **Comments:**:
#  - A smooth RKHS has small norm for any given function, which indicates that any difference of input would not effect the output significantly. Thus the property of smoothness is a wanted feature for machine learning.
#  
#  
# ### Summary
# 
#  - A p.d. kernel can be regarded as inner-product in some hilbert space after embedding(from data space $\mathcal{X}$ to hilert space $\mathcal{H}$).
#  - A realization of this embedding is the RKHS, which is not restricted by the dataset nor the kernel function.
#  - The RKHS is a function space defined on $\mathcal{X}$.
#  - The norm of function in RKHS is related to its degree of smoothness, with respect to the metric defined by the kernel function.
#  
# 

# ###  Kernel Tricks
# 
# Intuition: The embedding from original data space $\mathcal{X}$ to a higher dimensional hilbert space $\mathcal{H}$ could be beneficial for a machine learning algorithm. For instance, for a linear regression classifier on a linear unseperable dataset, if the kernel tricks could map the dataset into a higher dimensional space where the dataset is linear separable, then the performance of such classifier is enhanced significantly. 
# 
# **Basic paradigm of kernel tricks:** 
# 1. Choose a mapping function $\phi: \mathcal{X} \to \mathcal{H}$. Noted that the target hilbert space could has infinite dimension.
# 2. Embbed the original labeled dataset $S\{x_i, y_i\}$ into $S^*\{\phi(x_i),y_i\}$, where $y$ denotes the label.
# 3. Train a linear predictor over $S^*$.
# 4. Given the test point x_{new}, predict the label with predictor: $S^*(\phi(x_{new}))$. ???
# 
# #### Kernel Tricks for Supervised Learning
# 
# Given the space of input $\mathcal{X}$ and space of output $\mathcal{Y}$, and the training dataset $\{(x_i, y_i)\}_{i=1}^N$, the goal is to learn a predictor f such that:
# 
# \begin{align*}
# \min_f \frac{1}{N}\sum_{i=1}^N loss(f(x_i), y_i) + \lambda\Omega(\|f\|_{\mathcal{H}}) 
# \end{align*}
# 
# For instance, if we choose MSE as loss function, l2 norm as norm function, linear model as predictor, then the algorithm is the logistric regression. 
# 
# The kernel tricks is mainly focus on the range for optimization search, or in other words, the function space for f. Since in the kernel methods, we learn the target function f from a RKHS $\mathcal{H}$, we start from $f \in \mathcal{H}$. However, the function space $\mathcal{H}$ is infinite, in practice we need to narrow down the search space by the representer theorem.
# 
# #### Representer Theorem
# 
# Let $\Phi: \mathcal{R}^{N+1} \to \mathcal{R}$ be a stricly increasing function, then the optimization problem:
# \begin{align*}
# min_{f \in \mathcal{H}} \Phi\left(f(x_1),\cdots,f(x_N),\|f\|_{\mathcal{H}}\right)
# \end{align*}
# has the repsentation form of:
# \begin{align*}
# f(x)=\sum_{i=1}^N \alpha_i K(x_i,x) = \sum_{i=1}^N \alpha_i K_{x_i}(x)
# \end{align*}
# In other words, the result function f lies in a subset of RKHS: $span\{K_{x_1}(x),\cdots,K_{x_N}(x)\}$. Notice that the subset is finite with the dimension of N. Therefore, in practice we only need to search the finite subset for the optimal solution. 
# 
# **Proof:** TBC.
# 

# #### Least Square Regression with general function space 
# 
# 
# ### Kernel Ridge Regression (KRR)
# 
# Given a kernel function K, train the coefficient vector $\alpha \in \mathcal{R}^N$ by solving the optimization problem:
# 
# \begin{align*}
# \min_f \frac{1}{N}\sum_{i=1}^N loss(f(x_i),y_i) + \lambda \|f\|_{\mathcal{H}}^2 
# \end{align*}
# 
# By the representer theorem, any solution should takes the form of $f^*=\sum_{i=1}^N \alpha_i K_{x_i}(x)$.
# 
# 
# #### KRR in matrix form
# 
# Suppose K is the similarity matrix of kernel function, such that $K_{ij}=K(x_i,x_j)$. 
