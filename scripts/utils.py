
# coding: utf-8

# In[ ]:

import sys
import cv2
import numpy as np

# caffe layers
caffe_root = '/users/vijay.kumar/caffe/'
sys.path.insert(0, caffe_root + 'python')

sys.path.insert(0, '/users/vijay.kumar/tools/liblinear-2.1/python')
from liblinearutil import *

import caffe
from caffe import layers as L

def read_params():
    params = {}
    params['FEATSIZE'] = (4*4096) + 2000
    params['ids_h'] = np.concatenate((np.array(range(4096)),np.array(range(2*4096, 3*4096))), axis=0)        
    params['ids_u'] = np.concatenate((np.array(range(4096,2*4096)),np.array(range(3*4096,4*4096))), axis=0)

    return params 

def define_transformer():
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': (1,3,227,227)})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    return transformer

def load_models():
    
    nets = {}
    proto_file = '../models/deploy.prototxt'    
    
    for model_no in range(-1,7,1):       
        
        if model_no==-1:
            weights = '../models/base.caffemodel'
        else:
            weights = '../models/pose' + str(model_no) + '.caffemodel'            
        nets[model_no+1] = caffe.Net(proto_file, weights, caffe.TEST)
        nets[model_no+1].forward(start='conv1')                          

    pose_proto_file = '../models/pose_estimator.prototxt'
    pose_weights = '../models/pose_estimator.caffemodel'

    pose_net = caffe.Net(pose_proto_file, pose_weights, caffe.TEST)
    pose_net.forward()
    
    return nets, pose_net


def get_pose_features(transformer, nets, head, upper_body, num_m, feat_size):

    img_feat = np.zeros((num_m, feat_size))
    for model_no in range(num_m):
        
        net = nets[model_no]        
        img_feat[model_no] = None
        for i in range(2):            
            if i==1:
                head = cv2.flip(head,1)
                upper_body = cv2.flip(upper_body,1)
            transformed_image = transformer.preprocess('data', head)
            transformed_image2 = transformer.preprocess('data', upper_body)
            net.blobs['head'].data[0, ...] = transformed_image
            net.blobs['ub'].data[0, ...] = transformed_image2
            net.forward(start='conv1')
            
            feat = np.concatenate((net.blobs['fc6'].data, net.blobs['fc6_p'].data,
                                   net.blobs['conc_fc7'].data, net.blobs['fc7_plus'].data), 
                                  axis=1)            
           
            if i == 0:
                img_feat[model_no] = 0.5*feat.reshape(-1)
            else:
                img_feat[model_no] += 0.5*feat.reshape(-1)
            
    return img_feat
    
    
def get_pose_weights(transformer, net, upper_body):
            
    transformed_image = transformer.preprocess('data', upper_body)
    net.blobs['ub'].data[0, ...] = transformed_image
    net.forward(start='conv1')
    sc = net.blobs['fc8_pose'].data[0]   
    sc = sc/np.linalg.norm(sc)
    sc = (sc - np.min(sc))/(np.max(sc) - np.min(sc))        
    return sc


# crop head/upper body images from IMDB trainset
def get_region_imdb(image, box, region_type):
    
    region = None
    box = box.astype(int)
    if region_type == 'HEAD':
        region = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2],:]
                                
    if region_type == 'UB':
        l = min(box[2], box[3])
        ub_box_x = int(box[0] - 0.5*l)
        ub_box_y = int(box[1])
        ub_box_w = int(2*l)
        ub_box_h = int(4*l)

        region = image[max(0, ub_box_y): min(ub_box_y + ub_box_h, image.shape[0]),
                                    max(0, ub_box_x): min(ub_box_x + ub_box_w, image.shape[1]), :]                               
    return region

# crop head/upper body images from Hannah movie dataset
def get_region_hannah(image, fbox, region_type):
    
    region = None
    fbox = fbox.astype(int)
    if region_type == 'HEAD':
        fbox[1] = fbox[1]-0.5*fbox[3]
        fbox[3] = 1.7*fbox[3]
        fbox[0] = fbox[0] - 0.2*fbox[2]
        fbox[2] = 1.4*fbox[2]
        region = image[max(1,fbox[1]):min(fbox[1]+fbox[3], image.shape[0]), 
                    max(1,fbox[0]):min(fbox[0]+fbox[2], image.shape[1]),:]            
        
    if region_type == 'UB':
        
        fbox[1] = fbox[1]-0.5*fbox[3]
        fbox[3] = 1.7*fbox[3]
        fbox[0] = fbox[0] - 0.2*fbox[2]
        fbox[2] = 1.4*fbox[2]  
        
        l = min(fbox[2], fbox[3])
        ub_box_x = int(fbox[0] - 0.5*l)
        ub_box_y = int(fbox[1])
        ub_box_w = int(2*l)
        ub_box_h = int(4*l)

        region = image[max(0, ub_box_y): min(ub_box_y + ub_box_h, image.shape[0]),
                                    max(0, ub_box_x): min(ub_box_x + ub_box_w, image.shape[1]), :]  
        
    return region


def train_linear_classifiers(train_features, train_labels, num_models, params):
    
    # Classifier training
    classifiers = {}
    ids_h = params['ids_h']
    ids_u = params['ids_u']
    for model_no in range(num_models):            

        train_model_features = np.squeeze(train_features[:,model_no,:])   
        classifiers[model_no] = {}
        classifiers[model_no][0] = None
        classifiers[model_no][1] = None
        classifiers[model_no][2]= None

        # head_feature    
        print 'Model:',model_no, 'training using head feature..'        
        classifiers[model_no][0] = train(train_labels.tolist(), 
                                    train_model_features[:,ids_h].tolist(), 
                                    '-s 1 -c 1 -q')

        # UB feature
        print 'Model:',model_no, 'training using UB feature..'        
        classifiers[model_no][1] = train(train_labels.tolist(), 
                                 train_model_features[:,ids_u].tolist(), 
                                 '-s 1 -c 1 -q')

        print 'Model:',model_no, 'training using joint feature..'
        classifiers[model_no][2] = train(train_labels.tolist(), 
                                  train_model_features.tolist(), 
                                  '-s 1 -c 1 -q')
    return classifiers


def pose_aware_identity_prediction_(classifiers, test_feature, test_label, pose_weights, params, num_models):
    
    predict_sc = None    
    ids_h = params['ids_h']
    ids_u = params['ids_u']
    for model_no in range(num_models):
        
        tf = test_feature[model_no].reshape(1,-1)
                
        # pose specific head model                            
        [ig1, ig2, sc_h] = predict(test_label.tolist(), tf[:,ids_h].tolist(), classifiers[model_no][0], '-q')    
        
        # pose specific UB model        
        [ig1, ig2, sc_ub] = predict([test_label], tf[:,ids_u].tolist(), classifiers[model_no][1], '-q')    
    
        # pose specific head-UB model    
        [ig1, ig2, sc_h_ub] = predict([test_label], tf.tolist(), classifiers[model_no][2], '-q')    
        
        if model_no ==0:
            pose_sc = 1
        else:
            pose_sc = pose_weights[model_no-1]

        if model_no == 0:        
            predict_sc = pose_sc*(np.array(sc_h) + np.array(sc_ub) + np.array(sc_h_ub))
        else:
            predict_sc += pose_sc*(np.array(sc_h) + np.array(sc_ub) + np.array(sc_h_ub))
            
    return predict_sc    

