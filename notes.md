
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
