import numpy as np
import os
import os.path
import tkinter as tk

from PIL import Image, ImageDraw

from dataset import Dataset, freeman_from_np_2d, load_image_into_2d, image_to_np_2d, IMAGE_SIZE
from DT import DT
from knn import Knn
from naive_bayes import NaiveBayes

class DigitRecognizerApp:
    def __init__(self, win):
        self.win = win
        self.canvas = tk.Canvas(self.win, width=300, height=300, bg="black", bd=2, relief=tk.GROOVE)
        self.canvas.grid(row=0, column=0)

        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_digit)

        self.clear_button = tk.Button(self.win, text="Effacer", command=self.clear_digit)
        self.clear_button.grid(row=1, column=0)

        self.predict_button = tk.Button(self.win, text="Predire", command=self.predict)
        self.predict_button.grid(row=2, column=0)

        # Results
        self.nb_prediction = tk.Label(self.win, text="Naive Bayes:")
        self.nb_prediction.grid(row=0, column=1)

        self.knn_prediction = tk.Label(self.win, text="KNN (K = 8):")
        self.knn_prediction.grid(row=1, column=1)

        self.dt_prediction = tk.Label(self.win, text="Decision Tree:")
        self.dt_prediction.grid(row=2, column=1)

    def draw_digit(self, event):
        x, y = event.x, event.y
        new_color = 255
        self.draw.rectangle([x, y, x + 5, y + 5], fill=new_color)
        self.canvas.create_rectangle(x, y, x + 5, y + 5, fill="white", outline="white")

    def clear_digit(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        resized_image = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
        image_arr = image_to_np_2d(resized_image)
        print(image_arr.shape)
        print(image_arr)

def main():
    # np.set_printoptions(threshold=np.inf)
    print("Loading dataset...")
    # dataset = Dataset("ImageMl")
    print("Dataset loaded.")

    window = tk.Tk()
    window.title("Digit recognizer")
    window.geometry("400x500")
    app = DigitRecognizerApp(window)
    window.mainloop()

if __name__ == "__main__":
    main()
