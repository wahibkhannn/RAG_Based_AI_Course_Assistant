import whisper

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/3.7 Check If A Student Is Pass Or Fail.mp3",
                    
                        task = "translate", word_timestamps = False)


print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})
print(chunks)

with open("transcripts/3.7 Check If A Student Is Pass Or Fail.txt", "w") as f:
    json.dump