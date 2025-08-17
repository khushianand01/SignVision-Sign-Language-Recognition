import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from keras.models import load_model
import time

# Load model
model = load_model("c/keras_model.h5") 

# List of class labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize GUIimport cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from keras.models import load_model
import time

# Load model
model = load_model("c/keras_model.h5")

# List of class labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize GUI
root = Tk()
root.title("ASL Interpreter")
root.geometry("800x600")

# Global variables
predicted_text = ""
previous_letter = ""
last_prediction_time = time.time()
confidence_threshold = 0.6
min_time_between_predictions = 3

# GUI Widgets
video_label = Label(root)
video_label.pack()

text_display = Text(root, height=5, width=50, font=("Helvetica", 16))
text_display.pack()

def save_text(event=None):
    with open("output.txt", "w") as f:
        f.write(predicted_text)

save_button = Button(root, text="Save to File", command=save_text)
save_button.pack()

# Start video capture
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    roi = frame[100:300, 100:300]  # Adjust based on where your hand will be
    roi = cv2.resize(roi, (224, 224))  # match the model's expected input
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

def update_frame():
    global previous_letter, predicted_text, last_prediction_time

    ret, frame = cap.read()
    if not ret:
        return

    # Draw a rectangle to show the ROI
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

    roi = preprocess_frame(frame)
    prediction = model.predict(roi)
    letter_index = np.argmax(prediction)
    confidence = prediction[0][letter_index]

    if confidence > confidence_threshold:  
        letter = labels[letter_index]

        # Avoid repeated letters within min_time_between_predictions seconds
        current_time = time.time()
        if letter != previous_letter or (current_time - last_prediction_time) > min_time_between_predictions:
            predicted_text += letter
            previous_letter = letter
            last_prediction_time = current_time

            # Update text area
            text_display.delete(1.0, END)
            text_display.insert(END, predicted_text)

    # Convert frame to ImageTk
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

def reset_text(event=None):
    global predicted_text, previous_letter
    predicted_text = ""
    previous_letter = ""
    text_display.delete(1.0, END)

def close_app(event=None):
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.bind('<q>', close_app)
root.bind('<r>', reset_text)
root.bind('<s>', save_text)

update_frame()
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()