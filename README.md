# Person recognition
This repository contains the implementation of paper "Pose-Aware Person Recognition" by Vijay Kumar, Anoop Namboodiri, Manohar Paluri, C V Jawahar published at CVPR17.

The implementation is based on Python Caffe.

Datasets:
1. Download the datasets from the below links and place in data/ folder.
2. PIPA (test): [Link](https://people.eecs.berkeley.edu/~nzhang/piper.html)
3. Hannah movie : [Link](http://www.technicolor.com/en/innovation/scientific-community/scientific-data-sharing/hannah-dataset)
4. IMDB : [Link](http://cvit.iiit.ac.in/images/Projects/PersonRecognition/Data/imdb.zip)
5. Soccer videos : [Link](http://cvit.iiit.ac.in/images/Projects/PersonRecognition/Data/soccer.zip)


Models:
1. Download the trained models and place in models/ folder.
2. The models (baseline, pose-specific and pose estimator) are available at [link](http://cvit.iiit.ac.in/images/Projects/PersonRecognition/models.zip)


Testing:

Dependencies: [Liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

1. To reproduce the results on PIPA test set, run run_PIPA.ipynb
2. For recognition in movie scenario, run run_hannah.ipynb
3. For recognition in soccer setting, run run_soccer.ipynb
4. Change the data folder variable in these scripts according to your path.
5. Replace the liblinear path to your correct liblinear installation directory.



References:

If you use this code or data, please cite our paper along with below related references.

@InProceedings{vijaycvpr15,
 
author    = "Vijay Kumar and Anoop Namboodiri and and Manohar Paluri and Jawahar, C.~V.",

title     = "Pose-Aware Person Recognition",

booktitle = "Proceedings of IEEE International Conference on Computer Vision and Pattern Recognition",

year      = "2017"

}

Related references:

1. N. Zhang et al., Beyond Fronta Faces: Improving Person Recognition using Multiple Cues, CVPR 2014.
2. Oh et al., Person Recognition in Personal Photo Collections, ICCV 2015.
3. Li et al., A Multi-lvel Contextual Model for Person Recognition in Photo Albums, CVPR 2016.
4. Ozerov et al., On Evaluating Face Tracks in Movies, ICIP 2013.
