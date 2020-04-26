import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import natsort
from PIL import Image


import hopenet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.', 
                        default='roi_extracted_frames', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='Directory path for saving embeddings.',
                        default='user_embeddings', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='../snapshot/hopenet_robust_alpha1.pkl', type=str)

    args = parser.parse_args()

    return args

def generate(frames_path, sample_save_path):
    '''
    args:
    frames_path: Path of video frames to be read.
    sample_save_path: Path to save the output embedding.
    '''
    sample_embedding = []
    for i, img_path in enumerate(natsort.natsorted(os.listdir(frames_path))):
        if(img_path[-3:] != 'jpg'):
            continue
        img_path = os.path.join(frames_path, img_path)
        img = Image.open(img_path)
        img = transform(img)
        img = Variable(img).cuda(gpu)
        img = img.unsqueeze(0)
                
        yaw, pitch, roll = model(img)

        # Applying softmax on the predictions.
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)
                
        final_features = torch.cat((yaw_predicted, pitch_predicted, roll_predicted), -1)
        final_features = final_features.cpu().detach().numpy()
        sample_embedding.append(final_features)
    sample_embedding = np.squeeze(np.asarray(sample_embedding))
    assert(not np.isnan(sample_embedding).all())
    np.save(os.path.join(sample_save_path, "embedding.npy"), sample_embedding)


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    #CenterCrop and Scale need to be checke
    transform = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print ('Ready to generate embeddings..')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    data_dir = args.data_dir
    save = args.save_dir
    
    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        save_user_path = os.path.join(save, user)
        for sample in sorted(os.listdir(user_path)):
            if(len(sample)!=4): #Only allow samples of type ('01w1', '02s4' etc.)
                continue
            frames_path = os.path.join(user_path, sample, 'view2')
            sample_save_path = os.path.join(save_user_path, sample)
            if not os.path.exists(sample_save_path):
                os.makedirs(sample_save_path)
            if os.path.exists(os.path.join(sample_save_path, "embedding.npy")):
                print(sample_save_path, " is done.")
                continue

            print(frames_path)
            generate(frames_path, sample_save_path)