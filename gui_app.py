import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageGrab
from tensorflow.keras.models import load_model

MODEL_PATH = "../models/model.h5"

# ---------------------------
# MNIST-Style Preprocessing 
# ---------------------------
def preprocess(img):

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert (MNIST has white digit on black background)
    img = cv2.bitwise_not(img)

    # Gaussian blur (reduce noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Auto crop center → remove extra borders
    thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Resize to 28×28
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    return img


# ---------------------------
# Main GUI Application 
# ---------------------------
class DigitGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Digit Recognition (Improved GUI)")
        self.root.geometry("420x550")
        self.root.configure(bg="#F5F5F5")

        # Load trained model
        self.model = load_model(MODEL_PATH)

        Label(root, text="Draw a Digit (0-9)", font=("Arial", 18, "bold"), bg="#F5F5F5").pack(pady=10)

        # Canvas for drawing
        self.canvas = Canvas(root, width=300, height=300, bg="black")
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        btn_frame = Frame(root, bg="#F5F5F5")
        btn_frame.pack(pady=10)

        Button(btn_frame, text="Predict", command=self.predict_digit,
               bg="green", fg="white", font=("Arial", 14), width=10).grid(row=0, column=0, padx=10)

        Button(btn_frame, text="Clear", command=self.clear_canvas,
               bg="red", fg="white", font=("Arial", 14), width=10).grid(row=0, column=1, padx=10)

        # Prediction result
        self.result_label = Label(root, text="Prediction: None", font=("Arial", 18, "bold"), bg="#F5F5F5")
        self.result_label.pack(pady=20)

    # Draw on canvas
    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")

    # Clear canvas
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Prediction: None")

    # Predict the drawn digit
    def predict_digit(self):

        # Capture canvas
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        img = ImageGrab.grab((x, y, x + w, y + h))
        img = np.array(img)

        # Preprocess image
        processed = preprocess(img)

        # Predict
        prediction = self.model.predict(processed)
        digit = np.argmax(prediction)
        conf = float(np.max(prediction) * 100)

        self.result_label.config(text=f"Prediction: {digit}  ({conf:.2f}%)")


# ---------------------------
# Run Application
# ---------------------------
if __name__ == "__main__":
    root = Tk()
    app = DigitGUI(root)
    root.mainloop()
