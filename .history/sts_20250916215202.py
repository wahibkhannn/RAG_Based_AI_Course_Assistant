import whisper

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/3.7 Check If A Student Is Pass Or Fail.mp3",
                        language = "hi",
                        task = "translate")

print(result)