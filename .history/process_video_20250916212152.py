# converts the videos to mp3
# import whisper
import os
import subprocess

files = os.listdir("videos")
print(files)

for file in files:
    print(file)
    tutorial_number = file.split(".")[0]
    print(tutorial_number)
    file_name = 
