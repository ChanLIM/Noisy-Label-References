# Noisy-Label-References

1. Noisy Data Generation Methods

- [Learning with Biased Complementary Labels](https://arxiv.org/pdf/1711.09535.pdf) (Yu, ECCV 2018)
 >Where Y and Ybar is true and complementary labels, previous methods implicitly assume that 
 P(Y¯ = i|Y = j), ∀i ≠ j are identical, which is not true in practice because humans are biased toward their own experience.(표범만 봤던 사람은 치타를 봐도 표범이라고 label함) Therefore the transition probabilities should be different.
 
>Uses **complementary label** which specifies a class that an object does not belong to. Complementary labels are sometimes easily obtainable, especially when the class set is relatively large. Given an observation in multi-class classifcation, identifying a class label that is incorrect for the observation is often much easier than identifying the true label.

>Method : 확실히 아닌 하나를 빼고 나머지 9개에 대해 1)uniform probability 2)without 0 (3그룹으로 나누어서 합이 1이되게끔 0.2 0.1 0.033)
3)with 0 (3개 label 골라서 합이 1이 되게끔)

>related to [Learning from Complementary Labels](https://arxiv.org/pdf/1705.07541.pdf) (Ishida, NIPS 2017)

- [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/pdf/1804.06872.pdf) (Han, NIPS 2018)
 > uses pair flipping, symmetric flipping. Pair flipping refers to a case where a certain label is misclassified to a certain label since it's similar(but doesn't imply similarity in a way that two classes are paired). Symmetric flipping refers to a case where a label is not identified, so it is given any other random label
 
 > pair flipping method is not realistic in a way that two labels are just matched randomly, not according to how similar they look like so that people might make mistakes.
 
 
- [TRAINING DEEP NEURAL-NETWORKS USING A NOISE ADAPTATION LAYER](https://openreview.net/pdf?id=H12GRgcxg) (Goldberger, ICLR 2017)
 > Case 1: noisy labels are only dependent on the correct labels
 > Case 2: noisy labels are dependent on the features in addition to the correct labels


2. Related Works



3. Evaluation Measure
