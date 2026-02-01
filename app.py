import datetime #No need to add in requiremets
from flask import Flask, render_template, request, after_this_request
import time
from redmail import EmailSender
from werkzeug.utils import secure_filename #No need to add in requiremets
import os #No need to add in requiremets
import tensorflow as tf
import cv2 
import numpy as np
from dotenv import load_dotenv #No need to add in requiremets
from functools import reduce #No need to add in requiremets
load_dotenv()

last_video_path = None
INPUT_SIZE = (180, 180)
UPLOAD_FOLDER = 'static/videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load already learnt model
model = tf.keras.models.load_model("accident_detection_model.keras")
print("Model loaded Succefully")

app = Flask(__name__)

@app.route('/')
def home():    
    global last_video_path
    if last_video_path and os.path.exists(last_video_path):
        try:
            os.remove(last_video_path)
            print("Deleted on reload:", last_video_path)
        except Exception as e:
            print("Cleanup error:", e)
        finally:
            last_video_path = None

    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global last_video_path
    
    # delete previous video first
    if last_video_path and os.path.exists(last_video_path):
        try:
            os.remove(last_video_path)
            print("Deleted old video:",last_video_path)
        except Exception as e:
            print("Delete error: ",e)

    video_path = None
    output = None

    if request.method == 'POST':
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        videofile = request.files['photage']
        recipients_from_form = request.form.get('recipients')
        
        fname = secure_filename(videofile.filename)
        video_path = os.path.join(UPLOAD_FOLDER, fname)
        last_video_path = video_path
        videofile.save(video_path)
        request.video_path = video_path

        recipients = [email.strip() for email in recipients_from_form.split(',') if email.strip()]

        result = predict_accident(video_path, recipients, latitude, longitude)
        if type(result) != dict:
            output = result
        else:
            output = f"{(list(result.keys())[0])} with {(list(result.values())[0]) * 100:.1f}% chance" if list(result.keys())[0] == "🚨 Accident Detected" else f"{(list(result.keys())[0])}"

    return render_template('index.html', file_path=video_path, out=output)

#Prediction by taking video as input
def predict_accident(video_path, recipients, latitude, longitude):
    predictions = preprocess_video(video_path)
    if type(predictions) != list:
        return predictions

    #Binary Classification
    avg_pred = np.mean(predictions)
    label = "🚨 Accident Detected" if avg_pred > 0.5 else "✅ No Accident"

    if label == "🚨 Accident Detected":
        send_alerts(float(avg_pred), recipients, latitude, longitude)

    return {label : float(avg_pred)}
    
#Preprocess the input video
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate

    predictions =[]
    batch = []

    for sec in range(int(duration)):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success:
            break

        #preprocess frame
        frame = cv2.resize(frame, INPUT_SIZE)
        frame = frame / 255.0 # Normalize
        batch.append(frame[np.newaxis, :, :, :])

        if len(batch) == 5:
            batch_np = np.vstack(batch)
            probs = model.predict(batch_np, verbose=0).flatten()

            predictions.extend(probs)

            # 🚨 EARLY STOP
            if np.max(probs) >= 0.9:
                print("Early stop (batch)")
                break

            batch.clear()
    
    cap.release()
    return predictions
    
#trigger emails
def send_alerts(avg_pred, recipients, lt, lg):
    subject = f"🚨 Accident Detected | {datetime.datetime.now().strftime('%H:%M:%S')}"

    map_link = ""

    if lt and lg:
        map_link = f"https://www.google.com/maps?q={lt},{lg}"

    dt = list(datetime.datetime.now().ctime().split(" "))
    if dt[2] == "":
        time = dt.pop(4)
        date = reduce(lambda x,y : x + " " + y, dt)
    else:
        time = dt.pop(3)
        date = reduce(lambda x,y : x + " " + y, dt)

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif;">
        <h2 style="color:red;">An accident has been detected</h2>
        <p><strong>📊 Probability:</strong> {avg_pred * 100:.1f}%</p>
        <p><strong>⏱️ Time:</strong> {time.strip()}</p>
        <p><strong>📅 Date:</strong> {date.strip()}</p>
        <p><strong>📍 Location:</strong><a href="{map_link}" target="_blank"> Google Maps Link</a></p>
        <hr style="border:none;height:1px;background:#ddd;">
        <h3 style="color:red;"><strong>Please respond immediately.</strong></h3>
    </body>
    </html>
    """
    send_mail(subject, html_body, recipients)

#Send emails
def send_mail(subject, html_body, recipients):

    email = EmailSender(
        host="smtp.gmail.com",
        port=587,
        username=os.getenv("GMAIL_USER"),
        password=os.getenv("GMAIL_APP_PASSWORD"),
        use_starttls=True
    )

    email.send(
        subject=subject,
        receivers=recipients,
        html=html_body,
        text="This is a real-time accident alert."
    )

if __name__ == "__main__" :
    app.run(debug=True)
