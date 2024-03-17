import pandas as pd
import numpy as np
import face_recognition
from PIL import Image
import os, sys
from tqdm import tqdm

def crop_image(images_path):
    image_list = [image for image in os.listdir(images_path) if image.lower().endswith('jpg') or image.lower().endswith('png')]

    cropped_path = os.path.join(images_path, 'cropped')
    os.makedirs(cropped_path, exist_ok=True)
    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        image = Image.open(image_path)
        np_image = np.asarray(image)
        face_locations = face_recognition.face_locations(np.asarray(np_image))
        for idx, face_location in enumerate(face_locations):
            a, b, c, d = face_location
            face_location = d, a, b, c
            cropped_image = image.crop(face_location)
            cropped_image.save(os.path.join(cropped_path, f'{image_name}_{idx+1}.png'))
    
def crop_all_images(data_dir):
    image_path_list = os.listdir(data_dir)
    for image_path in tqdm(image_path_list):
        crop_image(os.path.join(data_dir, image_path))


def get_name(youtube_id):
    YTINFO_PATH = '/home/j.soyeon/vit/camp/campvideo-data/data/matches/ytinfo.csv'
    df_ytinfo = pd.read_csv(YTINFO_PATH)

    name = df_ytinfo.loc[df_ytinfo["uid"] == youtube_id, "candidate"].item()
    print(name)

    return name

def match_faces(youtube_id):
    REFERENCE_PATH = '/home/j.soyeon/vit/camp/campvideo-data/data/references'

def analyze_keyframes():
    pass

# for the future
if False:
    dta_path = '/home/j.soyeon/vit/camp/campvideo-data/data/wmp/wmp-senate-2012-v1.1.dta'
    # wmp-senate-2014-v1.0.dta

    YT_PATH = '/home/j.soyeon/vit/camp/campvideo-data/data/matches/ytinfo.csv'
    FRAMES_PATH = '/home/j.soyeon/vit/camp/campvideo-data/data/extracted_frames'

    df = pd.read_stata(dta_path, index_col='creative')
    # print(df.head(10))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # print(df.columns.to_list()) # o_picture, f_picture, vid exist

    df_input = df[['cand_id', 'category', 'o_picture', 'f_picture', 'vid', 'vidfile']]
    # print(df_input.head(3))
    print(df_input['cand_id'].sort_index().head(10))

    


if __name__ == '__main__':
    # sample_images_path = '/home/j.soyeon/vit/camp/campvideo-data/data/extracted_frames/'
    # crop_all_images(sample_images_path)

    get_name('_xqGsow5paI')