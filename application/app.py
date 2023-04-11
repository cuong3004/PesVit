# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import ttk
import tkinter as tk
import onnxruntime as ort
import datetime
import time 

# Create an instance of TKinter Window or frame
win= Tk()
win.title('PESTICIDE MISUSE DETECTION')

# Set the size of the window
win.geometry("700x350")# Create a Label to capture the Video frames
label0 = Label(win, text="PESTICIDE MISUSE DETECTION", fg="#1d2bba",font = ('Helvetica', 20, 'bold'))
label0.place(x=150, y=20)

label_show = Label(win)
label_show.place(x=100, y=180)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

model = ort.InferenceSession("model.onnx")

# label0 = Label(win, text="Video path",font = ('Helvetica', 12,))
# label0.place(x=100, y=100)

class Timer:
    def __init__(self) -> None:
        self.time_start = 0
    def start(self):
        self.time_start = time.time()
    def estimate(self):
        return time.time() - self.time_start

def writefile():
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open("time.txt", "w") as f:
        f.write(time_str)

    print("Saved time:", time_str)

def readfile():
    with open("time.txt", "r") as f:
        saved_time_str = f.read().strip()
    
    saved_time = datetime.datetime.strptime(saved_time_str, "%Y-%m-%d %H:%M:%S")
    now = datetime.datetime.now()
    time_diff = now - saved_time
    print("Days passed:", time_diff.days)
    return time_diff.days
    

timer = Timer()
timer.start()
agreement = tk.IntVar()

class Camera:
    def cam_func(self):
        if agreement.get() == 0:
            return
        if agreement.get() == 1:
            print(agreement.get())
            self.cap = cv2.VideoCapture(0)

            self.count_sau = 0
            self.label = 0

            def stream_camera():
                # Đọc dữ liệu từ camera
                ret, frame = cap.read()
                
                frame = cv2.resize(frame, (224, 224))
                input_frame = np.copy(frame)[None]
                cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                cv2image[:5] = np.zeros((5,224,3), dtype=np.uint8)
                cv2image[-5:] = np.zeros((5,224,3), dtype=np.uint8)
                cv2image[:,:5] = np.zeros((224,5,3), dtype=np.uint8)
                cv2image[:, -5:] = np.zeros((224,5,3), dtype=np.uint8)
                img = Image.fromarray(cv2image)

                image = input_frame / 255.0
                image = (image - mean) / std
                image = image.astype(np.float32)
                # print(image.dtype)

                image = np.transpose(image, (0, 3, 1, 2))

                # input_names = [input.name for input in model.get_inputs()]
                # print("Input names: ", input_names)

                outputs = model.run(None, {'input.1': image})[0]
                softmax_outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
                # global count_sau
                print(softmax_outputs) 
                # softmax_outputs = [[0,0.7]]
                time.sleep(0.1)
                if softmax_outputs[0][1] > 0.6:
                    self.count_sau += 1
                    show_predict_1()
                else:
                    self.count_sau = 0
                    show_predict_0()
                    

                if self.count_sau > 10:
                    writefile()
                    self.count_sau = 0
                
                if timer.estimate() > 30:
                    time_diff = readfile()
                    if time_diff > 10:
                        self.label = 0
                        show_label_0()
                    elif time_diff >= 7:
                        self.label = 1
                        show_label_1()
                    elif time_diff >= 0:
                        self.label = 2
                        show_label_2()
                    else:
                        raise "Error time diff < 0"
                    
                    timer.start()
                    print(time_diff)

                    
                # print(outputs)
                
                imgtk = ImageTk.PhotoImage(image = img)
                label_show.imgtk = imgtk
                label_show.configure(image=imgtk)

                

                # Lặp lại quá trình stream camera
                win.after(15, stream_camera)
            stream_camera()

def get_time_diff_video(a, b):
    # time_taken_str = entry_taken.get()
    # time_check_str = entry_check.get()
    time_diff = b - a
    return time_diff.days

class Video:
    def video_func(self):
        if agreement.get() == 0:
            return
        if agreement.get() == 2:
            print(agreement.get())
            self.cap = cv2.VideoCapture(entry_path.get())
            # self.cap = cv2.VideoCapture(0)

            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.count_sau = 0
            self.label = 0
            self.time_sau = 0

            self.issau = 0
            
            self.sau_time = datetime.datetime.strptime(entry_taken.get(), "%d/%m/%y")
            self.check_time = datetime.datetime.strptime(entry_check.get(), "%d/%m/%y")

            self.number_frame = 0
            def stream_camera():
                # Đọc dữ liệu từ camera
                ret, frame = self.cap.read()
                # frame = cv2.imread("1533175423-may-phun-thuoc-thoi-gio-2_jpg.rf.5f14c0ca428c89d0f27b85c50853521e.jpg")
                frame = cv2.resize(frame, (224, 224))
                input_frame = np.copy(frame)[None]
                cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                cv2image[:5] = np.zeros((5,224,3), dtype=np.uint8)
                cv2image[-5:] = np.zeros((5,224,3), dtype=np.uint8)
                cv2image[:,:5] = np.zeros((224,5,3), dtype=np.uint8)
                cv2image[:, -5:] = np.zeros((224,5,3), dtype=np.uint8)
                img = Image.fromarray(cv2image)

                image = input_frame / 255.0
                image = (image - mean) / std
                image = image.astype(np.float32)
                # print(image.dtype)

                image = np.transpose(image, (0, 3, 1, 2))

                # input_names = [input.name for input in model.get_inputs()]
                # print("Input names: ", input_names)

                outputs = model.run(None, {'input.1': image})[0]
                softmax_outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
                # global count_sau
                print(softmax_outputs) 
                # softmax_outputs = [[0,0.7]]
                # time.sleep(0.1)
                if softmax_outputs[0][1] > 0.6:
                    show_predict_1()
                    self.count_sau += 1
                else:
                    self.count_sau = 0
                    show_predict_0()

                if self.count_sau :
                    # writefile()
                    self.sau_time += datetime.timedelta(seconds=int(self.number_frame/self.fps))
                    self.count_sau = 0
                    self.issau = 1
                    self.number_frame = 0

                if timer.estimate() > 10 and self.issau:
                    time_diff = get_time_diff_video(self.sau_time, self.check_time)
                    
                    
                    if time_diff > 10:
                        self.label = 0
                        show_label_0()
                    elif time_diff >= 7:
                        self.label = 1
                        show_label_1()
                    elif time_diff >= 0:
                        self.label = 2
                        show_label_2()
                    else:
                        raise "Error time diff < 0"
                    
                    timer.start()
                    print(time_diff)

                    
                # print(outputs)
                
                imgtk = ImageTk.PhotoImage(image = img)
                label_show.imgtk = imgtk
                label_show.configure(image=imgtk)

                self.number_frame += 1
                

                # Lặp lại quá trình stream camera
                win.after(15, stream_camera)
            stream_camera()
# def video_func():
#     if agreement.get() == 0:
#          return
#     if agreement.get() == 1:
#         print(agreement.get())
    

checkbut1 = Checkbutton(win,
                text='From real-time camera',
                command=Camera().cam_func,
                variable=agreement,
                onvalue=1,
                offvalue=0)
checkbut1.place(x=100, y=80)

checkbut2 = Checkbutton(win,
                text='Video path',
                command=Video().video_func,
                variable=agreement,
                onvalue=2,
                offvalue=0)
checkbut2.place(x=100, y=100)

entry_path = Entry(win, width=30) 
entry_path.place(x=190, y=102)

def on_entry_click(event):
    if entry_taken.get() == "dd/mm/yy":
        entry_taken.delete(0, tk.END)
        entry_taken.config(fg='black')
    
    if entry_check.get() == "dd/mm/yy":
        entry_check.delete(0, tk.END)
        entry_check.config(fg='black')

label_taken = Label(win, text="Date of taken")
label_taken.place(x=120, y=130)
entry_taken = Entry(win) 
entry_taken.insert(0, "dd/mm/yy")
entry_taken.config(fg='grey')
entry_taken.bind('<FocusIn>', on_entry_click)
entry_taken.place(x=200, y=130)
label_check = Label(win, text="Date check")
label_check.place(x=350, y=130)
entry_check = Entry(win) 
entry_check.insert(0, "dd/mm/yy")
entry_check.config(fg='grey')
entry_check.bind('<FocusIn>', on_entry_click)
entry_check.place(x=420, y=130)



# frame = cv2.imread("1533175423-may-phun-thuoc-thoi-gio-2_jpg.rf.5f14c0ca428c89d0f27b85c50853521e.jpg")
# frame = cv2.resize(frame, (224, 224))
# img = Image.fromarray(frame)
# imgtk = ImageTk.PhotoImage(image = img)
# label_show.imgtk = imgtk
# label_show.configure(image=imgtk)


label_result = Label(win, text="Result prediction")
label_result.place(x=430, y=250)


label_hight = Label(win, text="Hight risk")
label_hight.place(x=450, y=275)
# label_hight.pack()
# label_hight.pack_forget()

label_low = Label(win, text="Low risk")
label_low.place(x=450, y=275)
# label_low.pack()
# label_low.pack_forget()

label_normal = Label(win, text="Normal", font = ('Helvetica', 12, 'bold'))
label_normal.place(x=440, y=280)
# label_normal.pack()
# label_normal.pack_forget()
def hidden_all_label():
    label_hight.config(text="")
    label_low.config(text="")
    label_normal.config(text="")

def show_label_0():
    # label_normal
    # label_hight.config(text="")
    # label_low.config(text="")
    label_normal.config(text="Normal")

def show_label_1():
    # label_hight.config(text="")
    # label_low.config(text="Low risk")
    label_normal.config(text="Low risk")
    
def show_label_2():
    # label_hight.config(text="Hight risk")
    # label_low.config(text="")
    label_normal.config(text="Hight risk")

hidden_all_label()
# show_label_0()
# label_normal.pack()


label_predict_normal = Label(win, text="Normal", fg="#01bd00", font = ('Helvetica', 12, 'bold'))
label_predict_normal.place(x=185, y=410)

label_predict_spray = Label(win, text="Spraying", font = ('Helvetica', 12, 'bold')) # , fg="#ff0000"
label_predict_spray.place(x=177, y=410)

label_predict_normal.config(text="")
label_predict_spray.config(text="")


def show_predict_0():
    # label_predict_normal.config(text="Normal")
    label_predict_spray.config(text="Normal")

def show_predict_1():
    # label_predict_normal.config(text="")
    label_predict_spray.config(text="Spraying")
# entry1 = Entry(win) 
# entry1.place(x=350, y=105)

# label6 = Label(win, text="Havest time")
# label6.place(x=390, y=140)
# label6.config(bg= "gray51")

# entry2 = Entry(win) 
# entry2.place(x=350, y=168)

# button1 = Button(win, text="Analysis")
# button1.place(x=393, y=200)
# label6.config(bg= "gray51")

# Define function to show frame
def show_frames():
      # Get the latest frame and convert into Image
      # frame = cap.read()[1]
      frame = cv2.imread("image.png")
      frame = cv2.resize(frame, (224, 224))
      # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      cv2image[:5] = np.zeros((5,224,3), dtype=np.uint8)
      cv2image[-5:] = np.zeros((5,224,3), dtype=np.uint8)
      cv2image[:,:5] = np.zeros((224,5,3), dtype=np.uint8)
      cv2image[:, -5:] = np.zeros((224,5,3), dtype=np.uint8)
      img = Image.fromarray(cv2image)

      # Convert image to PhotoImage
      imgtk = ImageTk.PhotoImage(image = img)
      label.imgtk = imgtk
      label.configure(image=imgtk)



# Repeat after an interval to capture continiously
#label.after(20, show_frames)

#show_frames()
win.mainloop()