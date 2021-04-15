import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#The mail addresses and password
sender_address = 'bishtshivam096@gmail.com'
sender_pass = 'Bisht@123456'
receiver_address = 'singhshipra07@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Employee Report'   #The subject line
#The body and the attachments for the mail



import time
initTime = time.time()
sumEmotion = [0,0,0,0,0,0,0]

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:

    startTime = time.time()
    # checkTime = int(startTime - initTime)%60

    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        sumEmotion[max_index]+=1

        if not int(startTime - initTime)%30:
            # print("Average emotion: ", emotions[sumEmotion.index(max(sumEmotion))])
            with open (f"mood.txt{time.time()}", "w") as f:
                f.write(f"Average mood of the employee is {emotions[sumEmotion.index(max(sumEmotion))]}")


            mail_content = ""
            mail_content = f'''Dear HR Manager,
Employee name: Shivam Singh Bisht
Average mood: {emotions[sumEmotion.index(max(sumEmotion))]}.
Working time : 30 sec'''

            message.attach(MIMEText(mail_content, 'plain'))

            #Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
            session.starttls() #enable security
            session.login(sender_address, sender_pass) #login with mail_id and password
            text = message.as_string()
            session.sendmail(sender_address, receiver_address, text)
            session.quit()
            print('Mail Sent')







            
            # Average mood of the employee is :: //mood
            initTime = 30 + time.time()
            sumEmotion = [0,0,0,0,0,0,0]



        

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows