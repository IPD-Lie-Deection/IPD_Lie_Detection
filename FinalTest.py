import subprocess
import os
import librosa
import numpy as np
import tensorflow as tf
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model

top_three_emotions = []

TF_ENABLE_ONEDNN_OPTS=0
TF_DISABLE_MKL=1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def Mp4toWav():
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
    Audio()

def Audio():
    

    RAVDE_CLASS_LABELS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")

    model_path = "E:\\IPD\\Neural_Customized\\Emotion-Classification-Ravdess\\Emotion_Voice_Detection_Model_87.h5"

    model = tf.keras.models.load_model(model_path)

    audio_file_path = 'E:\\IPD\\Neural_Customized\\Emotion-Classification-Ravdess\\03-01-04-01-02-02-01.wav'

    target_sample_rate = 16000      
    audio, sample_rate = librosa.load(audio_file_path, sr=target_sample_rate)
    n_mfcc = 1
    hop_length = 512
    n_fft = 2048

    mfccs = librosa.feature.mfcc(y=audio, sr=target_sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    max_time_steps = mfccs.shape[1]
    desired_time_steps = 40
    desired_num_features = 1

    if max_time_steps < desired_time_steps:
        pad_width = ((0, 0), (0, desired_time_steps - max_time_steps))
        mfccs = np.pad(mfccs, pad_width=pad_width, mode='constant')
    elif max_time_steps > desired_time_steps:
        mfccs = mfccs[:, :desired_time_steps]

    input_data = mfccs.T
    input_data = input_data.reshape(desired_time_steps, desired_num_features)
    input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    # predictions = predictions[:, 2:10]
    print(predictions)

    predicted_class = RAVDE_CLASS_LABELS[np.argmax(predictions)]
    print("Predicted Emotion:", predicted_class)
    audio_emotion=predicted_class
    video_emotions=top_three_emotions
    
    detect_deception_combined(audio_emotion, video_emotions)

    return predicted_class

def Video():
    
    model = load_model("E:\\IPD\\Neural_Customized\\Emotion-Classification-Ravdess\\model_1.h5",compile=False)

    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))


    emotion_count = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    total_frames = 0

    while True:
        ret, test_img = cap.read()  
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

           
            emotion_count[predicted_emotion] += 1

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        total_frames += 1

        
        out.write(test_img)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):  
            break

    cap.release()
    out.release()

    
    cv2.destroyAllWindows()

    
    dominant_emotion = max(emotion_count, key=emotion_count.get)
    print("Dominant Emotion:", dominant_emotion)

    
    sorted_emotions = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)[:3]

    print("Top Three Emotions:")
    for emotion, count in sorted_emotions:
        percentage = (count / total_frames) * 100
        print(f"{emotion}: {percentage:.2f}%") 
    

    for emotion, _ in sorted_emotions:
        top_three_emotions.append(emotion)
    
    Mp4toWav()
    
    return top_three_emotions



def detect_deception(emotions):

    combinations = [
        ["Angry", "Disgust", "Fear", "Sad"],
        ["Happy", "Neutral", "Sad", "Disgust"],
        ["Angry", "Fear", "Disgust", "Calm"],
        ["Happy", "Neutral", "Fear", "Sad"],      
    ]
    
    emotions.sort()
    
    for combo in combinations:
        
        sorted_combo = sorted(combo)
        if emotions == sorted_combo:
            return 1
    
    deception_subsets = [
        ["Angry", "Disgust", "Fear"],
        ["Happy", "Neutral", "Fear"],
        ["Angry", "Fear", "Disgust"],
        ["Neutral", "Fear", "Disgust"],
        ["Happy", "Neutral", "Fear"],
    ]
    
    
    for subset in deception_subsets:
        
        sorted_subset = sorted(subset)
        if emotions == sorted_subset:
            return 1
        
    return 0

def detect_deception_combined(audio_emotion, video_emotions):

    combined_emotions = [audio_emotion] + video_emotions
    common_emotion = None
    for emotion in combined_emotions:
        if combined_emotions.count(emotion) > 1:
            common_emotion = emotion
            break
    
    if common_emotion:
        combined_emotions.remove(common_emotion)
    if detect_deception(combined_emotions)==1:
        print("Lie")
    else:
        print("Truth")

Video()


