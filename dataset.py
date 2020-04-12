import numpy as np
import glob
import time
import cv2
import hp
import os
from torch.utils.data import Dataset
import pickle

#dictionary mapping words to list of cluster indexes.
vid_labels = {
    'old': [3, 6, 4],
    'family': [5, 4, 9, 3, 6, 2],
    'house': [5, 3, 2, 4, 1],
    'some': [4, 3, 9, 1],
    'single': [4, 3, 9, 5, 6, 1],
    'i am sorry': [3, 8, 4, 9, 8, 4, 3, 2, 2, 2],
    'best time to live': [9, 1, 4, 2, 8, 2, 3, 9, 1, 8, 2, 3, 8, 6, 3, 7, 1],
    'thank you': [2, 5, 4, 9, 6, 8, 2, 3, 2],
    'nice to meet you': [9, 3, 7, 1, 8, 2, 3, 8, 9, 1, 1, 2, 8, 2, 3, 2],
    'would': [1, 3, 2, 6, 4],
    'good bye': [5, 3, 3, 4, 8, 9, 2, 1],
    'ice': [3, 7, 1],
    'live': [6, 3, 7, 1],
    'take': [2, 4, 6, 1],
    'method': [9, 1, 2, 5, 3, 4],
    'listen': [6, 3, 4, 2, 1, 9],
    'locate': [6, 3, 7, 4, 2, 1],
    'fly': [5, 6, 2],
    'how are you': [5, 3, 1, 8, 4, 2, 1, 8, 2, 3, 2],
    'leg': [6, 1, 5],
    'he will forget it': [5, 1, 8, 1, 3, 6, 6, 8, 5, 3, 2, 5, 1, 2, 8, 3, 2],
    'catch the trade winds': [7, 4, 2, 7, 5, 8, 2, 5, 1, 8, 2, 2, 4, 4, 1, 8, 1, 3, 9, 4, 4],
    'come': [7, 3, 9, 1],
    'box': [9, 3, 7],
    'i never gave up': [3, 8, 9, 1, 7, 1, 2, 8, 5, 4, 7, 1, 8, 2, 3],
    'excuse me': [1, 7, 7, 2, 4, 1, 8, 9, 1],
    'work': [1, 3, 2, 6],
    'have a good time': [5, 4, 7, 1, 8, 4, 8, 5, 3, 3, 4, 8, 2, 3, 9, 1],
    'large': [6, 4, 2, 5, 1],
    'hear a voice within you': [5, 1, 4, 2, 8, 4, 8, 7, 3, 3, 7, 1, 8, 1, 3, 2, 5, 3, 9, 8, 2, 3, 2],
    'place': [3, 6, 4, 7, 1],
    'learn': [6, 1, 4, 2, 9],
    'see you': [4, 1, 1, 8, 2, 3, 2],
    'you are welcome': [2, 3, 2, 8, 4, 2, 1, 8, 1, 1, 6, 7, 3, 9, 1],
    'hello': [5, 1, 6, 6, 3]
}


file_to_word = {
    '01w': 'locate',
    '02w': 'single',
    '03w': 'family',
    '04w': 'would',
    '05w': 'place',
    '06w': 'large',
    '07w': 'work',
    '08w': 'take',
    '09w': 'live',
    '10w': 'live',
    '11w': 'method',
    '12w': 'listen',
    '13w': 'house',
    '14w': 'learn',
    '15w': 'come',
    '16w': 'some',
    '17w': 'ice',
    '18w': 'old',
    '19w': 'fly',
    '20w': 'leg',
    
    '01p': 'hello',
    '02p': 'excuse me',
    '03p': 'i am sorry',
    '04p': 'thank you',
    '05p': 'good bye',
    '06p': 'see you',
    '07p': 'nice to meet you',
    '08p': 'you are welcome',
    '09p': 'how are you',
    '10p': 'have a good time',
    
    '01s': 'i never gave up',
    '02s': 'best time to live',
    '03s': 'catch the trade winds',
    '04s': 'hear a voice within you',
    '05s': 'he will forget it'
}


def load_file(filename, m_embedding):
    """
    Load numpy file contained embedding of video frames and
    uniformly selecting every hp.sample_frames out of all frames.
    Sampling rate is dropped if input length is less than target length
    (due to ctc loss).
    
    Arguments:
        String: filename of the numpy file.
        Int: length of the target cluster sequences.
    Returns:
        arr: ([frames sampled , embedding_dim])
        length: number of frames sampled 
    """
    v_embedding = np.load(filename)
    arr = []
    for i in range(0,v_embedding.shape[0],hp.sample_frames):
        arr.append(v_embedding[i])
    
    #always include ending frame
    if (v_embedding.shape[0]-1)%hp.sample_frames!=0:
        arr.append(v_embedding[-1])
    
    v_embedding = np.array(arr)               
    
    max_len = max(v_embedding.shape[0],m_embedding.shape[0])
    embedding = np.zeros((max_len, v_embedding.shape[1]+m_embedding.shape[1]))
    for i in range(v_embedding.shape[0]):
                         embedding[i,:198] = v_embedding[i,:]
    for i in range(m_embedding.shape[0]):
                         embedding[i,198:] = m_embedding[i,:]
    length = embedding.shape[0]
    return embedding, length


    
class MyDataset(Dataset):
    def __init__(self, dataset,phase):
        stats_file = open(hp.stats, 'rb')
        stats = pickle.load(stats_file)
        self.mean = stats['mean']
        self.std = stats['std']
        muse_path = hp.data_path
        video_path = hp.video_data_path
        self.list = []
        self.len  = 0
        users = hp.dataset_split[phase]
        for user in users:
            samples = list(set(os.listdir(os.path.join(muse_path, user))) & set(os.listdir(os.path.join(video_path, user))))
            for sample in sorted(samples):
                word = file_to_word[sample[:3]]
                if not (phase =='train' and sample in hp.train_words or phase == 'val' and sample in hp.val_words or  phase == 'test' and sample in hp.test_words):
                    continue
                self.list.append((os.path.join(muse_path, user, sample), os.path.join(video_path, user, sample, 'view2'), vid_labels[word]))
                self.len+=1

    def __getitem__(self, idx):
        labels = self.list[idx][2]
        muse_item_path = self.list[idx][0]
        video_item_path = self.list[idx][1]
        muse_embed = np.load(os.path.join(muse_item_path, hp.features+'.npy'))
#        muse_embed = (muse_embed-self.mean)/(self.std + 1e-9)
        
        embedding, e_len = load_file(os.path.join(video_item_path, 'embedding.npy'),muse_embed)
        return embedding, e_len, labels, len(labels)
        
    def __len__(self):
        return self.len


if __name__ =='__main__':
    cnt =0; ln = 0
    for k in file_to_word:
        if(k[2]=='s'):
            cnt+=1
            ln+=len(file_to_word[k])
    print (cnt,ln)
    print (dataset[0][0].shape)
