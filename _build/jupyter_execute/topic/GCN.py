#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection
# 
# ## node2vec: Scalable Feature Learning for Networks
# 
# GOAL: node2vec, a semi-supervised algorithm for scalable feature learning in networks.
# 
# **Def. Link Prediction:** Predict whether a pair of nodes in a network should have an edge connecting them.
# 

# ## Applying support vector data description for fraud detection
# 
# Khedmati, Mohamad, Masoud Erfani, and Mohammad GhasemiGol. "Applying support vector data description for fraud detection." arXiv preprint arXiv:2006.00618 (2020).
# 
# **One-class Classification(OCC):** Identify objects of **a specific class amongst all objects**, by primarily learning from a training set containing **only the objects of that class**(not necessarily). It is more difficult than the traditional classification problem, which tries to distinguish between two or more classes with the training set containing objects from all the classes. [wiki](https://en.wikipedia.org/wiki/One-class_classification) 
# 
# **Support Vector Data Description(SVDD):** SVM based one-class classification, which relies on identifying the smallest hypersphere (with radius $r$, and center $c$) consisting of all the data points $x = (x_1,x_2,...,x_n)$. The problem can be defined as:
#  \begin{align*}
#  & \min_{r,c} r^2 \\
#  & s.t. \|\Phi(x_i) - c\| \le r^2, \forall i=1,2,...,n
#  \end{align*}
# However, the optimization problem given is highly restrictive and sensitive to outliers, since all the data points should be included in the hypersphere. To overcome the problem, we introduce slack variables $\zeta = (\zeta_1,\zeta_2,...,\zeta_n)$:
#  \begin{align*}
#  & \min_{r,c} r^2  + \frac{1}{n \nu} \sum_{i=1}^n \zeta_i \\
#  & s.t. \|\Phi(x_i) - c\| \le r^2 + \zeta_i, \forall i=1,2,...,n
#  \end{align*}
# Noted that the objective function should be penalized to restrict the slack variables. For the relaxed optimization problem, we apply the KKT conditions.
# 
# **Density-based spatial clustering of applications with noise(DBSCAN):**
# The DBSCAN algorithm can be abstracted into the following steps: [wiki](https://en.wikipedia.org/wiki/DBSCAN)
#  - Find the points in the $\epsilon$-neighborhood of every point, and identify the core points with more than minPts neighbors.
#  - Find the connected components of core points on the neighbor graph, ignoring all non-core points.
#  - Assign each non-core point to a nearby cluster if the cluster is an $\epsilon$-neighbor, otherwise assign it to noise.
# 
# Proposed Algorithm:
#  1. One-class Classification: we ONLY need non-fraud data to train a SVDD model! 
#  2. Sample Reduction(REDBSCAN): since SVDD is based on boundaries, the data samples could be reduced as long as the "shape" remains the same. 
#  3. Tune hyper-parameter based on Genectic Algorithm: NOT required, other method such as NN is also suitable.
#  
#  
# 
# 

# ## SynchroTrap: Uncovering Large Groups of Active Malicious Accounts in Online Social Networks
# 
# **Def. Jaccard Similarity:** The Jaccard similarity measures the similarity between two sets of data, which is calculated by dividing the number of observations in both sets by the number of observations in either set. 
# 
# \begin{align*}
# Sim(A, B) = \frac{|A \cap B|}{|A \cup B|}
# \end{align*}
# 
# **Def. Account Similarity in SynchroTrap:** The similarity between two accounts is based on the Jaccard Similarity. Let a time-stamped user action be denoted as a tuple $(U_i, T_i, C_i)$, representing the user-id, action timestamp and the constraint object. Note that the constraint is defined by the its application, such as ip, fingerprint and even a combination of these entities. Then we define that two actions "match"($\approx$) if they share the exact same constraint and fall into the same time window with a pre-defined length $T_{sim}$:
# \begin{align*}
# (U_i, T_i, C_i) \approx (U_j, T_j, C_j)
# \end{align*}
# if $C_i = C_j$ and $|T_i - T_j| \le T_{sim}$.
# 
#  - Per-constraint Similarity: the similarity between users i & j on constraint k is defined as
#  \begin{align*}
#  Sim(Ui, Uj, C_k) = \frac{|A_i^k \cap A_j^k|}{|A_i^k \cup A_j^k|}
#  \end{align*}
#  where set $A_i^k = \{(U, T, C) \mid U=U_i, C=C_k\}$, and the intersection and union is caculated by the operation of "match" we defined previously.
#  
#  - Overall Similarity: the overall similarity between users i & j is defined as
#  \begin{align*}
#  Sim(Ui, Uj) = \frac{\sum_k |A_i^k \cap A_j^k|}{\sum_k |A_i^k \cup A_j^k|}
#  \end{align*}
#  Note that the overall similarity is NOT based on per-constraint similarity. It applies to applications where each constraint could be used only once(such as app-install).
# 
# 
# **Def. Single-linkage:** Single-linkage clustering is one of several methods of hierarchical clustering. It is based on grouping clusters in **bottom-up fashion**, at each step combining two clusters that contain the closest pair of elements not yet belonging to the same cluster as each other.
# 
# 
# **Def. Scalable User Clustering:** The user clustering algorithm is based on the bottom-up fashion single-linkage clustering. The algorithm uses an agglomerative approach which begins with each user as a different cluster, and iterately merge clusters with high similarity to produce large clusters. In addition, to make the algorithm scalable for parallel implementation, the algorithm is simplified as the connected components in the pruned user similarity graph filtered with a given similarity threshold. 
# 
