## Our custom loss

![Loss](./Loss.jpeg)

* We initial all ωk as 1 then the loss can be seen as a ranking loss.

* To avoid the difference of two unequal depth values being too large and ease the problem of imbalanced ordinal relations,
    * we first sort the loss of unequal pairs at each iteration,
    * and then ignore the smallest part by setting corresponding ωk to 0.

* More specifically, we empirically set the smallest 25% of ωk to 0.
* Therefore, the ratio of equal relation would be increased so that the problem of imbalanced ordinal relations can be alleviated.
* In addition, the ConvNet is thus enforced to focus on a set of hard pairs during training.

