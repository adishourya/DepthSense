
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

