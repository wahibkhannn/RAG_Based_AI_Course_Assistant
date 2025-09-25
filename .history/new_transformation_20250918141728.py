import whisper
import json
import os

# Load the Whisper model
model = whisper.load_model("large-v2")

# Paths
audio_folder = "audios"
output_folder = "transcripts"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through files in the audio folder
# for file_name in os.listdir(audio_folder):
#     # Check if file name starts with 6.1 to 6.9
#     if file_name.startswith(tuple([f"6.{i}" for i in range(1, 10)])) and file_name.endswith(".mp3"):
#         audio_path = os.path.join(audio_folder, file_name)
#         print(f"Processing: {file_name}")

#         # Transcribe the audio
#         result = model.transcribe(
#             audio=audio_path,
#             task="translate",
#             word_timestamps=False
#         )

#         # Extract chunks
#         chunks = []
#         for segment in result["segments"]:
#             chunks.append({
#                 "start": segment["start"],
#                 "end": segment["end"],
#                 "text": segment["text"]
#             })

#         # Save as JSON
#         output_path = os.path.join(output_folder, file_name.replace(".mp3", ".txt"))
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(chunks, f, ensure_ascii=False, indent=4)

#         print(f"Saved transcript to: {output_path}")

# print("All matching files processed!")
audio_files = os.listdir(audio_folder)
for audio in audio_files:
    if audio.startswith(tuple([f"6.{i}" for i in range(2, 3)])) and audio.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, audio)
        result = model.transcribe(
                audio=audio_path,
                task="translate",
                word_timestamps=False
                             )
        chunks=[]
        for segment in result["segments"]:
            chunks.append({'start': segment['start'], "end": segment["end"], "text": segment["text"]})

        chunks_with_metadata = {"chunks": chunks, "full_text": result["text"]}

        output_path = os.path.join(output_folder, audio.replace(".mp3", ".json"))
        with open(output_path, "w", encoding="utf-8") as f:
                 json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)
        print(f"Saved transcript to: {output_path}")