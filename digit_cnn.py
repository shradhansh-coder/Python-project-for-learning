# digit_project.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import ImageGrab
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

MODEL_PATH = "digit_recognition_model.h5"

# ----- for Train & Save Model -----
def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n🚀 Training started (will take time depending on your machine)...")
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    print("✅ Model saved as", MODEL_PATH)

# ----- helper: ensure model exists -----
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. You need to train first (option 1).")
        return False
    return True

# ----- 2) Predict from custom image (Method 1) -----
def predict_custom_image():
    if not ensure_model(): return
    model = load_model(MODEL_PATH, compile=False)
    path = input("Enter image path (e.g. C:/Users/You/Downloads/digit.png): ").strip().replace('"','').replace("'",'')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Image not found or invalid path.")
        return
    try:
        img = cv2.resize(img, (28, 28))
    except Exception as e:
        print("❌ Could not resize image:", e); return
    img = 255 - img            # invert to have white digit on black background
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    pred = np.argmax(model.predict(img))
    print("✅ Predicted Digit:", pred)
    # show image
    plt.imshow(img.reshape(28,28), cmap='gray'); plt.title(f"Prediction: {pred}"); plt.show()

# ----- 3) Predict from MNIST test dataset (Method 2) -----
def predict_from_test():
    if not ensure_model(): return
    model = load_model(MODEL_PATH, compile=False)
    (_, _), (x_test, y_test) = mnist.load_data()
    try:
        x_test_norm = x_test / 255.0
        x_test_norm = x_test_norm.reshape(-1,28,28,1)
    except Exception as e:
        print("❌ Data error:", e); return
    index_input = input("Enter test index (0–9999) or 'r' for random: ").strip()
    if index_input.lower() == 'r':
        import random
        idx = random.randint(0, len(x_test_norm)-1)
    else:
        try:
            idx = int(index_input)
        except:
            print("❌ Enter a valid integer index."); return
    if idx < 0 or idx >= len(x_test_norm):
        print("❌ Index out of range (0–9999).")
        return
    sample = x_test_norm[idx].reshape(1,28,28,1)
    pred = np.argmax(model.predict(sample))
    print("✅ Predicted:", pred, "| Actual label:", int(y_test[idx]))
    plt.imshow(x_test[idx], cmap='gray'); plt.title(f"Prediction: {pred} | Actual: {int(y_test[idx])}"); plt.show()

# ----- 4) GUI: Draw & Predict -----
def open_gui():
    if not ensure_model(): return
    model = load_model(MODEL_PATH, compile=False)

    window = tk.Tk()
    window.title("Digit Recognition - Draw & Predict")
    # make window not too small
    window.geometry("320x380")
    canvas = Canvas(window, width=280, height=280, bg='black')
    canvas.pack(pady=10)

    def draw(event):
        x,y = event.x, event.y
        r = 8
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')

    canvas.bind("<B1-Motion>", draw)

    def predict_digit():
        # capture canvas area
        x = window.winfo_rootx() + canvas.winfo_x()
        y = window.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        img = ImageGrab.grab().crop((x,y,x1,y1)).convert('L')
        img = img.resize((28,28))
        arr = 255 - np.array(img)    # invert
        arr = arr / 255.0
        arr = arr.reshape(1,28,28,1)
        res = np.argmax(model.predict(arr))
        label.config(text=f"Prediction: {res}")

    def clear_canvas():
        canvas.delete("all")
        label.config(text="Prediction: ")

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=6)
    Button(btn_frame, text="Predict", command=predict_digit, width=10, bg='green', fg='white').pack(side='left', padx=6)
    Button(btn_frame, text="Clear", command=clear_canvas, width=10, bg='red', fg='white').pack(side='left', padx=6)

    label = Label(window, text="Prediction: ", font=("Arial", 14))
    label.pack(pady=4)

    window.mainloop()

# ----- Main menu -----
def main_menu():
    print("\n=== Digit Recognition Using CNN ===")
    print("1 -> Train & Save Model")
    print("2 -> Predict from Custom Image (Method 1)")
    print("3 -> Predict from MNIST Test Dataset (Method 2)")
    print("4 -> Open GUI (Draw & Predict)")
    print("5 -> Exit")
    choice = input("Enter choice (1-5): ").strip()
    if choice == '1':
        train_model()
    elif choice == '2':
        predict_custom_image()
    elif choice == '3':
        predict_from_test()
    elif choice == '4':
        open_gui()
    elif choice == '5':
        print("Exiting.")
        return
    else:
        print("Invalid choice.")
    # after action, return to menu
    main_menu()

if __name__ == "__main__":
    main_menu()
