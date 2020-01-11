import os
import numpy as np
import pandas as pd 

"""
This code is used to calculate average gesture entry rate per minute for each user of user.
"""

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

embedding_path = '../../../user_embeddings'
frame_rate = 30
data = []
print ("total users are {}".format(len(os.listdir(embedding_path))))

#for each of the user
for i,user in enumerate(os.listdir(embedding_path)):
    print ("{}. {}".format(i+1,user))
    user_path = os.path.join(embedding_path,user)
    duration_for_user = 0.
    cluster_for_user = 0.
    #for each of the 105 sequence
    for i,seq in enumerate(os.listdir(user_path)):
        file_path = os.path.join(user_path,seq,'view2/embedding.npy')
        num_frames = np.load(file_path).shape[0]
        sec = num_frames/frame_rate  #duration of recording in seconds.
        _seq = file_to_word[seq[:-1]]
        seq_length = len(_seq)  #number of letter (clusters) in the sequence.
        print ("character represenetation : {} | length : {}".format(_seq,seq_length))
        
        cluster_for_user += seq_length
        duration_for_user += sec
        
    duration_for_user/=60 # sec to muinutes
    gesture_entry_rate = cluster_for_user/duration_for_user
    data.append([user,gesture_entry_rate])
    
df = pd.DataFrame(data, columns = ['Name', 'Gesture Entry Rate'])

df.to_csv('gesture_rate.csv')
    