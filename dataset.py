import numpy as np
import glob
import time
import cv2
import hp
import os
from torch.utils.data import Dataset

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


def load_file(filename,label_len):
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
    embedding = np.load(filename)
#     sample_frames = hp.sample_frames
#     while(1):
#         n_frames = (embedding.shape[0]-1)/sample_frames + 1
#         if (embedding.shape[0]-1)%sample_frames!=0:
#             n_frames += 1
#         if n_frames>=label_len:
#             break
#         sample_frames-=1
        
    arr = []
    for i in range(0,embedding.shape[0],hp.sample_frames):
        arr.append(embedding[i])
    
    #always include ending frame
    if (embedding.shape[0]-1)%hp.sample_frames!=0:
        arr.append(embedding[-1])
    
    length = len(arr)
    arr = np.array(arr)               
    
    return arr, length

class MyDataset(Dataset):

    def __init__(self, dataset,phase):
        path = hp.data_path
        self.list = []
        self.len  = 0
        for person in dataset:
            for f in sorted(os.listdir(path+'/'+person)):
                word = file_to_word[f[:3]]
                if not (phase =='train' and f in hp.train_words or phase == 'val' and f in hp.val_words or  phase == 'test' and f in hp.test_words):
                    continue
                    
                self.list.append((path+'/'+person+'/'+f,vid_labels[word]))
                self.len+=1

    def __getitem__(self, idx):
        labels = self.list[idx][1]
        item_path = self.list[idx][0]
        inputs, inputs_len  = load_file(item_path+'/freq_features.npy',len(labels))
        assert not np.isnan(inputs).any(), "{}".format(sum(np.isnan(inputs)))
        inputs = (inputs-np.mean(inputs, axis=0))/(np.std(inputs, axis=0) + 1e-5)
        assert not np.isnan(inputs).any(), "{}".format(sum(np.isnan(inputs)))
        assert inputs_len>=len(labels), " input seq is shorter. total frames-{} for {}".format(inputs.shape[0],item_path)
        return inputs, inputs_len, labels, len(labels)
        
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
