# Noisy-Label-References

## 1. Noisy Data Generation Methods

### - [Learning with Biased Complementary Labels](https://arxiv.org/pdf/1711.09535.pdf) (Yu, ECCV 2018)
 >Where Y and Ybar is true and complementary labels, previous methods implicitly assume that 
 P(Y¯ = i|Y = j), ∀i ≠ j are identical, which is not true in practice because humans are biased toward their own experience.(표범만 봤던 사람은 치타를 봐도 표범이라고 label함) Therefore the transition probabilities should be different.
 
>Uses **complementary label** which specifies a class that an object does not belong to. Complementary labels are sometimes easily obtainable, especially when the class set is relatively large. Given an observation in multi-class classifcation, identifying a class label that is incorrect for the observation is often much easier than identifying the true label.

>**Method** : 확실히 아닌 하나를 빼고 나머지 9개에 대해 1)uniform probability 2)without 0 (3그룹으로 나누어서 합이 1이되게끔 0.2 0.1 0.033)
3)with 0 (3개 label 골라서 합이 1이 되게끔)

>related to [Learning from Complementary Labels](https://arxiv.org/pdf/1705.07541.pdf) (Ishida, NIPS 2017)

### - [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/pdf/1804.06872.pdf) (Han, NIPS 2018)
 > Uses *pair flipping* and *symmetric flipping*. Pair flipping refers to a case where a certain label is misclassified to a certain label since it's similar(but doesn't imply similarity in a way that two classes are paired). Symmetric flipping refers to a case where a label is not identified, so it is given any other random label
 
 > pair flipping method is not realistic in a way that two labels are just matched randomly, not according to how similar they look like so that people might make mistakes.
 
 followed the noise generation method used in \
 [Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach](http://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf) (Patrini, CVPR 2017)
 and\
 [Training Deep Neural Networks on Noisy Labels with Bootstrapping](https://arxiv.org/pdf/1412.6596.pdf) (Reed, ICLR 2015)
 
 Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach - asymmetric, class-conditional noise, where each label y in the training set is flipped to ytilda while feature vectors are untouched. The noise transition matrix is row-stochastic and not necessarily symmetric across the classes.
 [Patrini, CVPR 2017](https://github.com/giorgiop/loss-correction/blob/master/noise.py) 
 - def noisify_mnist_asymmetric()\
 mistakes:\
        1 <- 7    automobile <- truck\
        2 -> 7    bird -> airplane\
        3 -> 8    deer -> horse\
        5 <-> 6   cat <-> dog
        
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n
        
 
| |0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
|0|  |   |   |   |   |   |   |   |   |   |
|1|  |   |   |   |   |   |   |   |   |   |
|2|  |   |1-n|   |   |   |   | n |   |   |
|3|  |   |   |1-n|   |   |   |   | n |   |
|4|  |   |   |   |   |   |   |   |   |   |
|5|  |   |   |   |   |1-n| n |   |   |   |
|6|  |   |   |   |   | n |1-n|   |   |   |
|7|  | n |   |   |   |   |   |1-n|   |   |
|8|  |   |   |   |   |   |   |   |   |   |
|9|  |   |   |   |   |   |   |   |   |   |

 
 
 
 
### - [TRAINING DEEP NEURAL-NETWORKS USING A NOISE ADAPTATION LAYER](https://openreview.net/pdf?id=H12GRgcxg) (Goldberger, ICLR 2017)
 > Case 1: noisy labels are only dependent on the correct labels
 
 > Case 2: noisy labels are dependent on the features in addition to the correct labels
 
 > 


## 2. Related Works



## 3. Evaluation Measure
