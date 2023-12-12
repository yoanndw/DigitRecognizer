import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw

# Compteurs du nombre d'images enregistrées par digits
digit_counters = {}

class DigitRecognitionApp:

    def __init__(self, fenetre):

        # Instancie la fenetre de l'interface utilisateur
        self.fenetre = fenetre
        self.fenetre.title("Le paint de la maison !")

        # Instancie la zone de dessin.
        self.canvas = Canvas(fenetre, width=300, height=300, bg="black", bd=2, relief=tk.GROOVE)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=10)

        # Instancie le bouton "Effacer" et l'associe à la fonction "effacer_dessin"
        self.bouton_effacer = Button(fenetre, text="Effacer", command=self.effacer_dessin)
        self.bouton_effacer.grid(row=10, column=0, padx=5, pady=10, sticky="ew")

        # Instancie le bouton "Enregistrer" et l'associe à la fonction "enregistrer_digit"
        self.bouton_enregistrer = Button(fenetre, text="Enregistrer", command=self.enregistrer_digit)
        self.bouton_enregistrer.grid(row=11, column=0, padx=5, pady=10, sticky="ew")

        # Boutons Digits
        self.digits_buttons = []
        self.selected_digit_button = None
        
        # Instancie les 10 boutons digits
        for i in range(10):
            bouton_digit = Button(fenetre, text=str(i), command=lambda i=i: self.selection_digit(i))
            bouton_digit.grid(row=i, column=1, padx=5, pady=5, sticky="ns")
            self.digits_buttons.append(bouton_digit)
            digit_counters[i] = 0

        self.dessiner = True
        self.selected_digit = None
        self.image = Image.new("L", (280, 280), color="white")
        self.draw = ImageDraw.Draw(self.image)

        # Associe le click gauche à la fonction "dessiner_digit"
        self.canvas.bind("<B1-Motion>", self.dessiner_digit)

    # Fonction de dessin.
    # Est activé lorsque l'utilisateur presse le click gauche de la souris.
    def dessiner_digit(self, event):
        if self.dessiner:
            x, y = event.x, event.y
            current_color = self.image.getpixel((x, y))
            new_color = 255 if current_color == 0 else 0
            self.image.putpixel((x, y), new_color)
            self.canvas.create_rectangle(x, y, x + 1, y + 1, fill="white", outline="white")

    # Fonction du bouton "Effacer". Remet tous les pixels en noirs.
    def effacer_dessin(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color="white")
        self.draw = ImageDraw.Draw(self.image)

    # Fonction du bouton "Enregistrer". 
    # Enregistre le digit sous le nom : digit selectionné _ numéro de l'image du digit
    def enregistrer_digit(self):
        global digit_counters
        if self.selected_digit is not None:
            self.image.save(f"AFAC/{self.selected_digit}_{digit_counters[self.selected_digit]}.png")
            digit_counters[self.selected_digit] += 1

    # Fonction des boutons "digit". Permet de selectionner le numéro du digit dessiné.
    # Sert à l'enregistrement. Sert d'annotation automatique.
    def selection_digit(self, digit):
        self.selected_digit = digit

        # Déselectionne l'ancien bouton digit.
        if self.selected_digit_button is not None:
            self.selected_digit_button.config(bg=self.fenetre.cget("bg"))

        # Sélectionne le nouveau bouton digit
        self.selected_digit_button = self.digits_buttons[digit]
        self.selected_digit_button.config(bg="blue")

if __name__ == "__main__":
    
    fenetre = tk.Tk()
    app = DigitRecognitionApp(fenetre)
    fenetre.geometry("400x450")
    fenetre.mainloop()
