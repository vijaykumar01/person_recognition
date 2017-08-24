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
1. To reproduce the results on PIPA test set, run run_PIPA.ipynb
2. For recognition in movie scenario, run run_hannah.ipynb
3. For recognition in soccer setting, run run_soccer.ipynb
4. Change the "data_path" variable in these scripts
