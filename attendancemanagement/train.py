import tkinter as tk 
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
window = tk.Tk()
window.geometry('1920x1080')
window.title("Face Recogniser")
window.configure(background='black')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System", bg="Green", fg="white", width=80,height=5)

message.place(x=500, y=40)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="black", bg="white")
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=30, bg="white", fg="black")
txt.place(x=650, y=200,height=35)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black", bg="white", height=2)
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=30, bg="white", fg="black")
txt2.place(x=650, y=300,height=35)

lbl3 = tk.Label(window, text="Status : ", width=20, fg="black", bg="white", height=2)
lbl3.place(x=400, y=400)

message = tk.Label(window, text="None", bg="white", fg="black", width=80, height=2, activebackground="white")
message.place(x=650, y=400)

lbl3 = tk.Label(window, text="Attendance : ", width=20, fg="black", bg="white", height=2)
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="", fg="black", bg="white", activeforeground="green", width=50, height=2)
message2.place(x=700, y=650)


def clear():
    txt.delete(0,'end')
    #message.configure(text="")


def clear2():
    txt2.delete(0, 'end')
    #message.configure(text="")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (txt.get())
    name = (txt2.get())

    if (is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        print("capturing images")

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
                message.configure(text="Capturing images...")

            if cv2.waitKey(100) and 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        message.configure(text="")
        res = "Images Saved for ID : " + Id + " Name : " + name+"  Successfully"
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name ,STATUS:FAILED"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id ,STATUS:FAILED"
            message.configure(text=res)
def TrainImages():
    message.configure(text="Training images...")
    print("Training images")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Images Trained Sucessfully"
    message.configure(text=res)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

clearButton = tk.Button(window, text="Clear", command=clear, fg="black", bg="white", width=20, height=2,
                        activebackground="Green")
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="black", bg="white", width=20, height=2,
                         activebackground="Green")
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="black", bg="white", width=20, height=2,
                    activebackground="Red")

takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="white", width=20, height=2,
                     activebackground="Red")
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Detect Images", fg="black", bg="white", width=20, height=2,
                     activebackground="Red")
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=2,
                       activebackground="Red")
quitWindow.place(x=1100, y=500)
window.mainloop()
