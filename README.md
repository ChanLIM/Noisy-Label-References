# Noisy-Label-References Abstract

<details>
  <summary>Show / Hide</summary>
  


</details>


## General Info, Survey Paper
<details>
  <summary>Show / Hide</summary>
  
[Learning with Label Noise Github Page](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise)

[Class Noise vs. Attribute Noise : A Quantitative Study of Their Impacts](http://www.cse.fau.edu/~xqzhu/papers/AIR.Zhu.2004.Noise.pdf)(Zhu, AI Review 2004)

According to the classification of label noise in \
[Classification in the Presence of Label Noise: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6685834)(Frenay, IEEE 2014)

</details>

## Taxonomy of Label Noise in the Survey Paper
<details>
  <summary>Show / Hide</summary>
  ### 1. NCAR - Noisy Completely at Random Model
> - the occurrence of an error E is independent of the other random variables, including the true class itself
> - biased coin of noise rate / fair dice to choose wrong label
> - uniform label noise

### 2. NAR - Noisy at Random Model
> - probability of error depends on the true class Y, but still independent of X
> - allows modeling asymmetric label noise, when instances from certain classes are more prone to be mislabeled.
> - NCAR label noise is a special case of NAR label noise.
> ex.) arbitrary labeling matrices & pairwise label noise
> - pairwise label noise : Two classes c1 and c2 are selected. Each instance of class c1 has a probability to be incorrectly labeled as c2 and vice versa. For this label noise, only two nondiagonal entries of the labeling matrix are nonzero.

NCAR and NAR considers that the label noise affects all instances with no distinction. -> not realistic\
Samples may be more likely mislabeled when they are similar to instances of another class.\
More difficult samples or low density (low encountered cases) may have higher chances of mislabeling.

### 3. NNAR - Noisy Not at Random Model
> - the occurrence of an error E is dependent on both variables X and Y,(mislabeling is more probable for certain classes and in certain regions of the X space.)
> - The most general case of label noise.
> - feature dependent한 경우(NNAR)와 feature independent한 경우(NCAR & NAR)의 경우로 나눌 수 있음.

</details>

According to the classification of label noise in \
[Classification in the Presence of Label Noise: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6685834)(Frenay, IEEE 2014)




## Noisy Data Generation Methods

### Objective : mimic the structure of noise in real life :
* mistakes for similar classes 
* mistakes for unknown classes

### - [Learning with Biased Complementary Labels](https://arxiv.org/pdf/1711.09535.pdf) (Yu, ECCV 2018)
 >Where Y and Ybar is true and complementary labels, previous methods implicitly assume that 
 P(Y¯ = i|Y = j), ∀i ≠ j are identical, which is not true in practice because humans are biased toward their own experience.(표범만 봤던 사람은 치타를 봐도 표범이라고 label함) Therefore the transition probabilities should be different.
 
>Uses **complementary label** which specifies a class that an object does not belong to. Complementary labels are sometimes easily obtainable, especially when the class set is relatively large. Given an observation in multi-class classifcation, identifying a class label that is incorrect for the observation is often much easier than identifying the true label.\
>(맞는 거 하나를 고르는 것보다 확실히 답이 아닌 하나를 고르는 labeling이 난이도가 낮음. 이를 이용해서 학습하려는 시도)

>**Method** : 확실히 틀린 class 하나를 빼고 나머지 9개에 대해 :

> 1. uniform probability
> 2. without 0 (3그룹으로 나누어서 합이 1이되게끔 0.2 0.1 0.033)
> 3. with 0 (3개 label 골라서 합이 1이 되게끔)

>related to [Learning from Complementary Labels](https://arxiv.org/pdf/1705.07541.pdf) (Ishida, NIPS 2017)
>label noise가 각 사람의 경험의 차이에 의해 많이 일어나는데, 이를 해결하기 위한 방안으로 complementary label 제시.
> -> true class에 영향을 받는 NAR

### - [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/pdf/1804.06872.pdf) (Han, NIPS 2018)
 > Uses **pair flipping** and **symmetric flipping**. Pair flipping refers to a case where a certain label is misclassified to a certain label since it's similar(but doesn't imply similarity in a way that two classes are paired). Symmetric flipping refers to a case where a label is not identified, so it is given any other random label
 
 > **How pair flipping is defined in this paper is different. Survey에서 말하는 pair flipping하고는 차이가 있음.** 
 > 'Coteaching pair flipping' method is not realistic in a way that two labels are just matched randomly, not according to how similar they look like so that people might make mistakes.
 > 개선 가능 지점.
 > label noise를 handling하는 많은 기법들이 NCAR, NAR 방법으로 만들어낸 noise 많이 사용을 하는데, 이는 사람이 만들어낸 noise랑은 차이가 많이 있을 수 있다. 이를 개선하기 위해 이런 종류의 noise모델을 제시한다.
 
 #### followed the noise generation method used in: 
> [Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach](http://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf) (Patrini, CVPR 2017) - asymmetric, class-conditional noise, where each label y in the training set is flipped to ytilda while feature vectors are untouched. The noise transition matrix is row-stochastic and not necessarily symmetric across the classes.
 [github codes](https://github.com/giorgiop/loss-correction/blob/master/noise.py) 
 >- def noisify_mnist_asymmetric()
 
        # 1 <- 7    (automobile <- truck / Some trucks are mistaken as automobile)
        # 2 -> 7    (bird -> airplane)
        # 3 -> 8    (deer -> horse)
        # 5 <-> 6   (cat <-> dog)
        
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n
        
 >Noise Transition Matrix, P
 >(Coteaching의 pair flipping을 일부 label에 적용한 경우에 해당)
 
| |0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
|0| 1|   |   |   |   |   |   |   |   |   |
|1|  | 1 |   |   |   |   |   |   |   |   |
|2|  |   |1-n|   |   |   |   | n |   |   |
|3|  |   |   |1-n|   |   |   |   | n |   |
|4|  |   |   |   | 1 |   |   |   |   |   |
|5|  |   |   |   |   |1-n| n |   |   |   |
|6|  |   |   |   |   | n |1-n|   |   |   |
|7|  | n |   |   |   |   |   |1-n|   |   |
|8|  |   |   |   |   |   |   |   | 1 |   |
|9|  |   |   |   |   |   |   |   |   | 1 |


CIFAR 100의 경우 Superclass 써서 조금 다른 방법
> 같은 superclass에 속하는 다른 class로 noise가 발생할 확률이 높다고 보고, superclass내의 class 끼리만 shuffle하는 방법.
> **NAR** - class pair 만들고 n의 확률로 서로에게 mapping해주는 pair-flipping. 
 
 
>[Training Deep Neural Networks on Noisy Labels with Bootstrapping](https://arxiv.org/pdf/1412.6596.pdf) (Reed, ICLR 2015)
>> Section 4.1 MNIST with Noisy Labels : Specifically, we used a **fixed random permutation** of the labels as visualized in figure 2, value on column is mapped to a value on row with some probability (didn't use CIFAR dataset)
 
        # 0 -> 2
        # 1 -> 5
        # 2 -> 4
        # ...

![Reed, ICLR 2015 Figure2](/img/Reed2015_Figure2.PNG)
 (original github code not available)\
->label을 sort해서 적용하면 coteaching pair flipping과 동일
>> **NAR**
 
>[Learning with Symmetric Label Noise: The Importance of Being Unhinged](https://arxiv.org/pdf/1505.07634.pdf) (van Rooyen, NIPS 2015)
 >>Symmetric label noise : where the learner observes samples from a distribution Dbar, which is a corruption of D where labels have some constant probability of being flipped. (Original Github Code Not Available)
 >> **NCAR**
 
 
 
### - [Training deep neural-networks using a noise adaptation layer](https://openreview.net/pdf?id=H12GRgcxg) (Goldberger, ICLR 2017)
 > Case 1: noisy labels are only dependent on the correct labels
 
 > Case 2: noisy labels are dependent on the features in addition to the correct labels
 
 
### [MentorNet]

### []
 
### - [Genre-based Decomposition of email class noise](http://delivery.acm.org/10.1145/1560000/1557070/p427-kolcz.pdf?ip=115.145.226.106&id=1557070&acc=ACTIVE%20SERVICE&key=0EC22F8658578FE1%2EB50D9BE1468BDDBD%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1569291337_66262b6ec3313a2ed5693aa62837814e)(Kolcz, ACM SIGKDD 2009)
 > Studies of data cleaning techniques often assume a uniform label noise model, however, which is seldom realized in practice.\
 > ... class noise can have substantial content specific bias. We also demonstrate that noise detection techniques based on classifier confidence tend to identify instances that human assessors are likely to label in error. 
>>**NNAR** 찾기 어려운데 이 논문의 경우 NNAR을 해결하기 위한 방법 제시.

그냥 inspiration 줄 만한 논문들
### - [Identifying Mislabeled Training Data](https://arxiv.org/pdf/1106.0219.pdf)(Brodley Friedl, JAIR 1999)
 > Using two kinds of filtering methods : consensus filters and majority vote filters
 > consensus filters - conservative at throwing away good data at the expense of retaining bad data
 > majority filters - better at detecting bad data at the expense of throwing away good data.


## Consequences of Label Noise on Learning
1. Theoretical and empirical evidences of impact of label noise on classification performances
2. Increases the necessary number of samples and complexity for learning
3. Distortion of observed frequencies
4. Deterioration of feature selection

## Approaches to Handle Label Noise
1. Label Noise-Robust Model
2. Label Cleansing Methods for Noisy Datasets
3. Label Noise-Tolerant Learning Model


## Evaluation Measure

1. Accuracy
2. Label Precision
