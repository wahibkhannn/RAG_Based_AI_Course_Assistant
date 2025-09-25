import whisper

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/1")