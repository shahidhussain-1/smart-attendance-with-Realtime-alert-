import cv2
import pandas as pd
from datetime import datetime
import pywhatkit
import pyautogui
import time
import os

# =========================
# LOAD TRAINED MODEL
# =========================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# ID → NAME MAPPING
# =========================
names = ["HUSSAIN", "varun"]

# =========================
# CAMERA START
# =========================
cam = cv2.VideoCapture(0)

present = []

# =========================
# CREATE CSV FILE IF NOT EXISTS
# =========================
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv("attendance.csv", index=False)

# =========================
# WHATSAPP FUNCTION (AUTO SEND)
# =========================
def send_whatsapp(phone, message):
    try:
        print("Sending message to:", phone)

        # Open WhatsApp Web and type message
        pywhatkit.sendwhatmsg_instantly(
            phone, message, wait_time=12, tab_close=False
        )

        # Wait for page to fully load
        time.sleep(10)

        # Click to ensure focus
        pyautogui.click()

        time.sleep(1)

        # Press ENTER automatically
        pyautogui.press('enter')

        print("Message sent automatically ✅")

        # Close tab after sending
        time.sleep(3)
        pyautogui.hotkey('ctrl', 'w')

    except Exception as e:
        print("Error sending message:", e)

print("Press 'q' to stop camera")

# =========================
# FACE RECOGNITION LOOP
# =========================
while True:
    ret, img = cam.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        print("Detected ID:", id, "Confidence:", conf)

        if conf < 70:
            if id < len(names):
                name = names[id]
            else:
                name = "Unknown"

            # Mark attendance only once
            if name not in present:
                present.append(name)

                now = datetime.now()
                time_now = now.strftime("%H:%M:%S")

                df = pd.DataFrame([[name, time_now]],
                                  columns=['Name', 'Time'])
                df.to_csv("attendance.csv", mode='a',
                          header=False, index=False)

        else:
            name = "Unknown"

        # Display name
        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (255, 0, 0), 2)

    cv2.putText(img, "Press Q to Exit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Attendance System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLOSE CAMERA
# =========================
cam.release()
cv2.destroyAllWindows()

# =========================
# ABSENT CHECK
# =========================
all_students = names[:]   # include all students
present = list(set(present))

absent = [s for s in all_students if s not in present]

print("Present:", present)
print("Absent:", absent)

# =========================
# SEND WHATSAPP ALERT
# =========================
phone = "+919010735139"

for student in absent:
    message = f"{student} is absent today"
    send_whatsapp(phone, message)

    # Wait between messages (VERY IMPORTANT)
    time.sleep(15)

print("All messages sent ✅")
