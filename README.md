# Noisy-Label-References

1. Noisy Data Generation Methods

- [Learning with Biased Complementary Labels](https://arxiv.org/pdf/1711.09535.pdf) (Yu, ECCV 2018)
 >Where Y and Ybar is true and complementary labels, previous methods implicitly assume that 
 P(Y¯ = i|Y = j), ∀i ≠ j are identical, which is not true in practice because humans are biased toward their own experience.(표범만 봤던 사람은 치타를 봐도 표범이라고 label함) Therefore the transition probabilities should be different.
 
>Uses **complementary label** which specifies a class that an object does not belong to. Complementary labels are sometimes easily obtainable, especially when the class set is relatively large. Given an observation in multi-class classifcation, identifying a class label that is incorrect for the observation is often much easier than identifying the true label.

- 

2. Related Works



3. Evaluation Measure
