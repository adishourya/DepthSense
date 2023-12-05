A project based exam for Deep learning for image and Video Programme
- Maastricht University
Last commit on main by 22nd Dec.
Best viewed in github/adishourya/DepthSense

# Depth Sense
    * Produces a reasonable depth map given a monocular iamge
    * Trained on RedWEbv1


## Methodology
* This project closely follows the practices of Andrej Karpathy's Micrograd series
* The util and pytorch board util scripts were inspired from the Dive into Deep Learning
* notes from the books can be found in notes/ folder

## Code Navigation :
Note that there is a code (report) pdf for every experiment , which goes into almost the detail you would need to understand the code
But the script for the file might slightly defer . Mostly to adjust shape of the tensors to match the requirements of the loss functions

    * Data Loaders :
        * ./data/loaders/data_loader_notebook.pdf: goes over making a simple dataset class with some utility classes like RandomCrop and Rescaling
        * ./data/loaders/OnlineLoader.pdf: goes over making a dataset where we generate ordinal distances between the region of the same image and treat the problem as a classic regression problem
        * The Readme inside data/loaders goes into more detail

    * Loss Function (spent about 90% of my 8 weeks here) :
        * Defintely checkout the Readme inside loss/
        * Each Loss has a notebook and a corresponding pdf which explains the loss calculation
        * ./online_sampling/OnlineSampling.pdf goes over one example of online samping calculation

    * Modelling :
        * ./models/training_loop.pdf goes over some epochs of how we trained the model
        * the full log of training epochs can be found in ./models/log.txt

    * Evaluation and results:


# References
from : Ke Xian1, Chunhua Shen2, Zhiguo Cao1*, Hao Lu1, Yang Xiao1, Ruibo Li1, Zhenbo Luo3
https://sites.google.com/site/redwebcvpr18/

# Monocular Relative Depth Perception with Web Stereo Data Supervision
Ke Xian1, Chunhua Shen2, Zhiguo Cao1*, Hao Lu1, Yang Xiao1, Ruibo Li1, Zhenbo Luo3
1School of Automation, Huazhong University of Science and Technology, China
2The University of Adelaide, Australia               3Samsung Research Beijing, China
e-mail: kexian@hust.edu.cn


# ReDWeb V1 dataset
The ReDWeb V1 dataset consists of 3.6K RGB-RD images, covering both indoor and outdoor scenes. Note that, this dataset can be used for research only.
Download (Please  feel free to reach out to me if you cannot download the ReDWeb dataset.)
BibTex
```
@inproceedings{Xian_2018_CVPR,
    title          = {Monocular Relative Depth Perception with Web Stereo Data Supervision},
    author     = {Xian, Ke and Shen, Chunhua and Cao, Zhiguo and Lu, Hao and Xiao, Yang and Li, Ruibo and Luo, Zhenbo},
     booktitle  = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year        = {2018}
}
```

Deep Dive into Deep Learning Book
```
@book{zhang2023dive,
    title={Dive into Deep Learning},
    author={Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
    publisher={Cambridge University Press},
    note={\url{https://D2L.ai}},
    year={223}
}
```

Video Lectures from Andrej Karpathy Micrograd Series
```
https://karpathy.ai
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
```
