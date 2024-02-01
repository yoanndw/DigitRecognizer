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
    def __init__(self, win, dataset: Dataset, naive_bayes: NaiveBayes, knn: Knn, dt: DT):
        self.dataset = dataset
        self.naive_bayes = naive_bayes
        self.knn = knn
        self.dt = dt

        self.init_ui(win)
    
    def init_ui(self, win):
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
        self.nb_prediction_var = tk.StringVar(self.win)
        self.nb_prediction_var.set("Naive Bayes:")
        self.knn_prediction_var = tk.StringVar(self.win)
        self.knn_prediction_var.set("KNN (K = 8):")
        self.dt_prediction_var = tk.StringVar(self.win)
        self.dt_prediction_var.set("Decision Tree:")

        self.nb_prediction = tk.Label(self.win, textvariable=self.nb_prediction_var)
        self.nb_prediction.grid(row=0, column=1)

        self.knn_prediction = tk.Label(self.win, textvariable=self.knn_prediction_var)
        self.knn_prediction.grid(row=1, column=1)

        self.dt_prediction = tk.Label(self.win, textvariable=self.dt_prediction_var)
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

        freeman = freeman_from_np_2d(image_arr)

        nb_pred = self.naive_bayes.predict(freeman)
        print("Naive Bayes:", nb_pred)
        self.nb_prediction_var.set(f"Naive Bayes: {nb_pred}")

        knn_pred = self.knn.predict(freeman)
        print("KNN (K = 8):", knn_pred)
        self.knn_prediction_var.set(f"KNN (K = 8): {knn_pred}")
        
        dt_pred = self.dt.predict(image_arr)
        print("Decision Tree:", dt_pred)
        self.dt_prediction_var.set(f"Decision Tree: {dt_pred}")


def main():
    # np.set_printoptions(threshold=np.inf)
    print("Loading dataset...")
    dataset = Dataset("ImageMl")
    print("Dataset loaded.")

    full_set = list(range(len(dataset.data)))
    print("Training models...")
    naive_bayes = NaiveBayes(dataset)
    knn = Knn(dataset, 8)
    dt = DT(dataset)

    naive_bayes.train(full_set)
    knn.train(full_set)
    dt.train(full_set)
    print("Trained models.")

    window = tk.Tk()
    window.title("Digit recognizer")
    window.geometry("400x500")
    app = DigitRecognizerApp(window, dataset, naive_bayes, knn, dt)
    window.mainloop()

if __name__ == "__main__":
    main()
