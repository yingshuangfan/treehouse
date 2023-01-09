#!/usr/bin/env python
# coding: utf-8

# # Graphic Convolutional Network
# 
# ## node2vec: Scalable Feature Learning for Networks
# 
# GOAL: node2vec, a semi-supervised algorithm for scalable feature learning in networks.
# 
# **Def. Link Prediction:** Predict whether a pair of nodes in a network should have an edge connecting them.
# 

# ## SynchroTrap: Uncovering Large Groups of Active Malicious Accounts in Online Social Networks
# 
# **Def. Jaccard Similarity:** The Jaccard similarity measures the similarity between two sets of data, which is calculated by dividing the number of observations in both sets by the number of observations in either set. 
# 
# \begin{align*}
# Sim(A, B) = \frac{|A \cap B|}{|A \cup B|}
# \end{align*}
# 
# **Def. Account Similarity in SynchroTrap:** The similarity between two accounts is based on the Jaccard Similarity. Let a time-stamped user action be denoted as a tuple $(U_i, T_i, C_i)$, representing the user-id, action timestamp and the constraint object. Note that the constraint is defined by the its application, such as ip, fingerprint and even a combination of these entities. Then we define that two actions "match"($approx$) if they share the exact same constraint and fall into the same time window with a pre-defined length $T_{sim}$:
# $$
# (U_i, T_i, C_i) \approx (U_j, T_j, C_j)
# $$
# if $C_i = C_j$ and $|T_i - T_j| \le T_{sim}$.
# 
#  - Per-constraint Similarity: the similarity between users i & j on constraint k is defined as
#  $$
#  Sim(Ui, Uj, C_k) = \frac{|A_i^k \cap A_j^k|}{|A_i^k \cup A_j^k|}
#  $$
#  where set $A_i^k = \{(U, T, C) \mid U=U_i, C=C_k\}$, and the intersection and union is caculated by the operation of "match" we defined previously.
#  
#  - Overall Similarity: the overall similarity between users i & j is defined as
#  $$
#  Sim(Ui, Uj) = \frac{\sum_k |A_i^k \cap A_j^k|}{\sum_k |A_i^k \cup A_j^k|}
#  $$
#  Note that the overall similarity is NOT based on per-constraint similarity. It applies to applications where each constraint could be used only once(such as app-install).
# 
# 
# **Def. Single-linkage:** Single-linkage clustering is one of several methods of hierarchical clustering. It is based on grouping clusters in bottom-up fashion, at each step combining two clusters that contain the closest pair of elements not yet belonging to the same cluster as each other.
# 
# 
# 
