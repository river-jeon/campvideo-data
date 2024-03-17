import yt_dlp
import os
from tqdm import tqdm
import sys
import pandas as pd
import csv

# urls = ['https://www.youtube.com/watch?v=yx4f2zK67q0']
option = {
        'outtmpl': './campvideo-data/data/videos/%(id)s.%(ext)s', 
        'format': 'bestvideo+bestaudio/best', # bestvideo/best: best quality video
        'postprocessors': [{  # Specify post-processing options
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'  # Ensure the final format is MP4
            }]
        } 

# read features.csv that contains url lists
df = pd.read_csv('./campvideo-data/data/features/features.csv')


# prepare url lists
df['url'] = 'https://www.youtube.com/watch?v=' + df['uid'] # prepend repeated url
urls = df['url']

with open('./error_files.csv', 'w') as f:
    writer = csv.writer(f)

    for url in urls: 
        try:
            with yt_dlp.YoutubeDL(option) as ydl:
                ydl.download([url]) # without audio

        except:
            print(f'error in {url}')
            writer.writerow([url])


