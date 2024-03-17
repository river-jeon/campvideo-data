# how many videos are missing 
from face_recognition_replicate import find_face
from extract_keyframes import extract_frames_ffmpeg
import pandas as pd 
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn

# read csv lines
data_df = pd.read_csv('./campvideo-data/data/features/features.csv')

print(f'Dataframe length before filtering election type: {len(data_df.index)}')

# data_df  USSEN/AK BEGICH WORK WITH ANYONE,_WtLKmT25_w,"47,63,98,139,156,196,213,249,264,304,333,354,409,434,476,519,582,606,676,712"
data_df = data_df[['creative','uid','keyframes']] # in order to select multiiple columns, use list (inner list)
data_df['election_type'] = data_df.apply(lambda x:x['creative'].split('/')[0], axis=1) # add into each row
data_df = data_df.loc[data_df['election_type'] == 'USSEN'] # loc filters only true rows
# data_df = data_df.loc[data_df['election_type'] == 'HOUSE'] # HOUSE/PA08 FITZPATRICK AS PARENTS
# data_df = data_df.loc[data_df['election_type'] == 'GOV'] # GOV/OH KASICH GREW UP

print(f'Dataframe length after filtering election type: {len(data_df.index)}')

data_df['name'] = data_df.apply(lambda x: x['creative'].split(' ')[1], axis=1)
data_df['state'] = data_df.apply(lambda x: x['creative'].split(' ')[0].split('/')[1], axis=1)


# read id names
id_list = os.listdir('./campvideo-data/data/ids')
# name_list = [lambda x:x.split('.')[0].split('_')[1:] for file_name in id_list] # name_list = [["NM", "udall"], ["NC", "hagan"]] [i for i in name_list][1]

def find_id(row, id_list):
    # find id
    for i in id_list:
        if i == 'obama.npy':
            continue
        if i.split('_')[1] in row['state'] and i.split('_')[2].split('.')[0].upper() in row['name']:
            return i

    return ''

data_df['ref_id_list'] = data_df.apply(lambda x:find_id(x, id_list), axis=1)
# print(data_df.head(5))

print(len(data_df.index)) # 769 non-missing videos
print(len([i for i in data_df.loc[data_df['ref_id_list'] == '']])) # 7 (missing videos)

frame_output_dir = './campvideo-data/data/extracted_frames'
video_dir = './campvideo-data/data/videos'

# create directories of frames
# maybe it's better to skip if file already exists?
results = []
for row in tqdm(data_df.iterrows()):

    uid_path = os.path.join(frame_output_dir, row['uid'])
    os.makedirs(uid_path, exist_ok = True)

    if os.listdir(uid_path) > 0:
        results.append(True)
    
    video_path =os.path.join(video_dir, f'{row["uid"]}.mp4')
    if not os.exists(video_path):
        results.append(False)
        continue
    keyframes = list(map(int, row['keyframes'].split(','))) if type(row['keyframes']) == str else row['keyframes']
    extract_frames_ffmpeg(video_path, keyframes, uid_path)

# filter those with frames

# real analysis begins here

# count # True
