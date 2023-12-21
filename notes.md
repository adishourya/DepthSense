
Mini-batch sampling Instead of training with fixed point pairs from each image
[6], we explore the diversity of sam- ples by online sampling, i.e., we resort
to sample pairs on- line within each mini-batch. For each input image I, we
randomly sample N point pairs (i,j), where N is the to- tal number of point
pairs, i and j represent the location of the first and second points,
respectively. To label ordinal relation lij between each point pair, we first
obtain depth values (gi , gj ) from corresponding ground-truth depth map, and
then define the ground-truth ordinal relation lij as follows:

lij = 1 if gi/gj > 1 + sigma , -1 if gi/gj < 1+ sigma , 0 otherwise

where ik and jk represent a location of the first and the second pair of point in the kth pair
second point in the k-th pair, and lk ∈ {+1, −1, 0} is the corresponding
ground-truth ordinal relationship between ik and jk that indicates further
(+1), closer (-1), and equal (0). Note that there exists the problem of
imbalanced ordinal relations, i.e., the number of equal relation is far less
than other two relations.

To enable our ConvNet to be trained with imbalanced ordinal relations, an appropriate loss function is needed. In this paper, we design an improved ranking loss L(I, G, z), which can be formulated as follows:
N
L(I, G, z) = sigma ωkφ(I, ik, jk, lk, z), (2)
k=1
where z is the estimated relative depth map, ωk and φ(I, ik, jk, lk, z) are the weight and loss of the k-th point

pair, respectively. Note that ωk can only be 0 or 1 in our experiments. φ(I, ik, jk, lk, z) takes the form:
φ = log(1 + exp[(−zik + zjk)lk]), lk ̸= 0, (3) (zik −zjk)2, lk =0.
We initial all ωk as 1, then the loss can be seen as a rank- ing loss [6]. To avoid the difference of two unequal depth values being too large and ease the problem of imbalanced ordinal relations, we first sort the loss of unequal pairs at each iteration, and then ignore the smallest part by setting corresponding ωk to 0. More specifically, we empirically set the smallest 25% of ωk to 0. Therefore, the ratio of equal relation would be increased so that the problem of imbalanced ordinal relations can be alleviated. In addition, the ConvNet is thus enforced to focus on a set of hard pairs during training.



This paper delves into the realm of monocular depth estimation, presenting a
novel approach that transforms the depth map estimation problem into a
regression framework. By treating depth prediction as a regression problem, our
proposed model aims to enhance the accuracy and efficiency of monocular depth
estimation. Moreover, the study investigates the impacts and functionalities of
various structure guidance loss techniques in refining the depth prediction.

Through extensive experimentation, we evaluate the performance of our
regression-based model across diverse datasets and real-world scenarios. We
analyze the nuanced effects of different structure guidance loss functions,
exploring their potential to enhance depth estimation precision and
generalization. The study provides insights into the interplay between
regression-based modeling and structure-guided methodologies, shedding light on
the mechanisms that contribute to improved depth map estimation.

Our findings showcase the potential of the proposed approach to advance the
state-of-the-art in monocular depth estimation, offering a deeper understanding
of the synergy between regression-based models and structure guidance losses.
This work contributes to the ongoing discourse on enhancing depth perception in
computer vision applications, with implications for fields such as autonomous
avigation, 3D reconstruction, and augmented reality.

Monocular depth estimation, although useful, is still a difficult and severely underconstrained problem to solve while still holding the interpreatbility of the model. It requires the use of numerous, occasionally subtle visual clues, distant context, and past knowledge to solve them. We require training data that reflects the diversity of the visual world and is equally varied in subject settings in order to develop models that perform well in a range of scenarios. 
The range and operating conditions of sensors that offer dense ground-truth depth in dynamic scenes, like time-of-flight or structured light, are constrained.In Stereo cameras we have left-right consistency which helps in calculating disparity of the subject by calculating pixel shift between 2 images . But a monocular dataset is more readily available and practical to hold . Thus quite a lot of work has went into estimating relative depth in monoclar images in the recent few years
