import subprocess
import os

input_dir = r"E:\IPD_GUI\WebCam-Face-Emotion-Detection-Streamlit"
ffmpeg_path =r"C:\Users\rihan\Downloads\ffmpeg\bin\ffmpeg.exe"

for root, dirs, files in os.walk(input_dir):
    for name in files:
        if name.endswith('.mp4'):
            input_file = os.path.join(root, name)
            output_file = os.path.join(root, name[:-4] + '.wav')
            
            command = [
                ffmpeg_path,
                '-i', input_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                output_file
            ]
            
            try:
                # Execute conversion using subprocess.Popen
                subprocess.Popen(command, shell=True)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
