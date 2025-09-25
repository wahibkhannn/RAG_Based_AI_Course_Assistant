# converts the videos to mp3
# import whisper
import os
import subprocess

files = os.listdir("videos")
print(files)

for file in files:
 
    # tutorial_number = file.split(".")[0]
  
    file_name = os.path.splitext(file)[0] #The [0] simply grabs the first element — the filename without the extension
    print(file_name)
    subprocess.run(['ffmpeg', "-i", f"videos/{file}", f"audios/{file_name}.mp3"])
