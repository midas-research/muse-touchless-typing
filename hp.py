import torch
import torch.nn as nn
import pickle
import hp
import random

num_classes = 10    # '_'(Blank) + 9 clusters
dec_label = ['1','2','3','4','5','6','7','8','9'] #clusters (decoding purpose)
display_interval = 1 #interval at which metrics for batches should be displayed.
sample_frames = 10
#constrained_beam_search = False #if beam search is restricted to specific sequences or not.
#trie = 'lexicon_35.pkl'
#allowed sequences in case of constrained beam search.
hp.lexicon = ['637421', '937', '1326', '364', '549362', '53348921', '53241', '4391', '51813668532512832', '439561', '3849843222', '91428239182386371', '742758251822441813944', '7391', '51663', '562', '64251', '389171285471823', '9371823891128232', '13264', '177241891', '615', '53184218232', '51428487337181325398232', '371', '5471848533482391', '254968232', '6371', '36471', '2461', '61429', '4118232', '232842181167391', '912534', '634219']

lr = 0.0025
hp.max_norm = 400 #Norm cutoff to prevent explosion of gradients
epochs = 200

#RNN related
inputDim = 198 + 60
hiddenDim =  512
rnn_layers = 4
batch_size = 20
bidirectional = True
rnn_type = nn.GRU
context = 20 #lookahead related (deepspeech)
beam_width = 100
#lrscheduler
sleep_epochs = 5
half = 50

model_base_path = './model'
data_path = '/media/data_dump_2/Pranay/dataset/main_preprocessing/museproject/dataset_muse' #folder contained numpy embedding matrix.
video_data_path = '../../user_embeddings'
model_path = '' #reloading file path
stats = '/media/data_dump_2/Pranay/dataset/main_preprocessing/museproject/sensor_preprocessing/mfcc_stats_f.pkl'
GPU = 0 #GPU ids available for use

features = 'mfcc_features'
#############EXPERIMENTATIONS###############################

#enter name of users in train and test respectively
split = {
    '1': ['aakash0910', 'abhishek2709', 'apoorv0209', 'siddharth2208', 'hitkul1808', 'lakshya0109'],
    '2': ['prakhar2308', 'sakshi0109', 'shahid3108', 'shivangi1908', 'aayush0209'],
    '3': ['anushka0110', 'shagun0409', 'harsh2509', 'ishaan0309', 'pakhi2408'],
    '4': ['ritwik0109',  'ashwat2508', 'shivam1808', 'gyanesh0909', 'suril2408']
}
val_set = ['apoorv0209']
train_set = ['prakhar2308', 'shagun0409', 'shahid3108', 'shivangi1908', 'aayush0209', 'sakshi0109', 'lakshya0109', 'hitkul1808', 'siddharth2208', 'abhishek2709', 'pakhi2408', 'aakash0910', 'ishaan0309', 'anushka0110', 'harsh2509']



dataset_split = {
    'train' : train_set,
    'val': val_set,
    'test' : split['4']
}

train_words= ['19w1', '19w2', '19w3', '12w1', '12w2', '12w3', '05p1', '05p2', '05p3', '13w1', '13w2', '13w3', '05s1', '05s2', '05s3', '01s1', '01s2', '01s3', '15w1', '15w2', '15w3', '16w1', '16w2', '16w3', '07p1', '07p2', '07p3', '09w1', '09w2', '09w3', '05w1', '05w2', '05w3', '02w1', '02w2', '02w3', '06p1', '06p2', '06p3', '08p1', '08p2', '08p3', '18w1', '18w2', '18w3', '14w1', '14w2', '14w3', '10w1', '10w2', '10w3', '07w1', '07w2', '07w3', '03s1', '03s2', '03s3', '03p1', '03p2', '03p3', '02p1', '02p2', '02p3', '20w1', '20w2', '20w3', '04p1', '04p2', '04p3', '06w1', '06w2', '06w3', '11w1', '11w2', '11w3', '03w1', '03w2', '03w3','04s1', '04s2', '04s3', '01p1', '01p2', '01p3', '02s1', '02s2', '02s3', '10p1', '10p2', '10p3', '09p1', '09p2', '09p3', '08w1', '08w2', '08w3', '01w1', '01w2', '01w3', '17w1', '17w2', '17w3', '04w1', '04w2', '04w3']
val_words = train_words
test_words = train_words








