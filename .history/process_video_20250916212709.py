# converts the videos to mp3
# import whisper
import os
import subprocess

files = os.listdir("videos")
print(files)

for file in files:
 
    # tutorial_number = file.split(".")[0]
  
    file_name = os.path.splitext(file)[0] + ".mp3"
    print(file_name)
    subprocess.run(['ffmpeg', "-i", file, f"{}"])
