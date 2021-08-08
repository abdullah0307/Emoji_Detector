#import tkinter as tk
from tkinter import *
import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(8, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(True)

emotion_dict = {0: "Angry", 1: "Disgusted",2: "Emoji" ,3: "Fearful", 4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprised"}
emoji_dist={0:"angry.png",2:"disgusted.png",3:"emoji.jpg",3:"fearful.png",4:"happy.png",5:"neutral.png",6:"sad.png",7:"surpriced.png"}
last_frame1 = np.zeros((600, 500, 3), dtype=np.uint8)
global cap1
filepath = ' '
show_text=[0]
def mfileopen():
	global filepath
	filepath=fd.askopenfile()

def show_vid(): 
	if(filepath!=' '):
		cap1=cv2.imread(filepath.name)  
		frame1 = cap1
		flag1 = cap1
		frame1 = cv2.resize(frame1,(600,500))
		print(filepath)
		bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

		for (x, y, w, h) in num_faces:
			cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
			roi_gray_frame = gray_frame[y:y + h, x:x + w]
			cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
			prediction = emotion_model.predict(cropped_img)
			
			maxindex = int(np.argmax(prediction))
			show_text[0]=maxindex
		if flag1 is None:
			print ("Major error!")
		else :
			global last_frame1
			last_frame1 = frame1.copy()
			pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
			print("WORKING") 
			img = Image.fromarray(pic)
			imgtk = ImageTk.PhotoImage(image=img)
			lmain.imgtk = imgtk
			lmain.configure(image=imgtk)
			lmain.after(10, show_vid)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			exit()


def show_vid2():
	if(filepath!=' '):
		frame2=cv2.imread(emoji_dist[show_text[0]])
		frame2 = cv2.resize(frame2,(300,400))
		pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
		img2=Image.fromarray(frame2)
		imgtk2=ImageTk.PhotoImage(image=img2)
		lmain2.imgtk2=imgtk2
		lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
		lmain2.configure(image=imgtk2)
		lmain2.after(10, show_vid2)

if __name__ == '__main__':
	print("Starting")
	root=Tk()
	img = ImageTk.PhotoImage(Image.open("log.jpg"))
	heading = Label(root,image=img,bg='black')
	
	
	width, height = root.winfo_screenwidth(), root.winfo_screenheight()  
	heading.pack()
	
	heading2=Label(root,text="EMOTICON",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
	SelectButton = Button(root,text="select photo",fg="red",command=mfileopen,width = 30).place(x=12,y=40)
	heading2.pack()
	lmain = Label(master=root,padx=50,bd=10,bg='black')
	lmain2 = Label(master=root,bd=10,bg='black')
	lmain3=Label(master=root,bd=10,fg="#CDCDCD",bg='black')
	lmain.pack(side=LEFT)
	lmain.place(x=30,y=250)
	lmain3.pack()
	lmain3.place(x=980,y=150)
	lmain2.pack(side=RIGHT)
	lmain2.place(x=1000,y=250)  
	root.title("Photo To Emoji")          
	root.geometry('%dx%d+0+0' % (width,height)) 
	root['bg']='black'
	DoneButton = Button(root,text='PREDICT',command=lambda:[show_vid(),show_vid2()]).place(x=250,y=40)
	exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).place(x=650,y=600)
	root.mainloop()
