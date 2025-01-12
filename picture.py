import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, UnidentifiedImageError



class View:
    def __init__(self, master):
        self.master = master
        self.frame()
        self.image = None  # variável para manter a referência para imagem 

        #Label para demonstrar emoções
        self.emotion_label = tk.Label(self.canvas, text="", font=('Arial', 16), bg="#FFFFFF", fg='#000000')
        self.canvas.create_window(200, 700, anchor='center', window=self.emotion_label)

    def frame(self):
        # Criação da frame
        self.master.attributes('-fullscreen', True)
        self.master.title('Facial Recognition')

        # Background
        self.canvas = tk.Canvas(self.master, width=1920, height=1080, highlightbackground='#333333', bg="#333333")
        self.canvas.place(x=0, y=0)

        # Cria um background para onde a imagem vai ser colocada
        self.canvas2 = tk.Canvas(self.master, width=400, height=550, highlightbackground='#333333', bg="#333333")
        self.canvas2.place(x=0, y=0)

        # Botão para o reconhecimento facial
        self.recon_button = tk.Button(self.canvas, text="Reconhecimento facial", command=self.read_emotion, font=('Arial', 14), bg='#d8e6f4')
        self.canvas.create_window(780, 620, anchor='center', window=self.recon_button)

        # Botão para fecho do programa
        self.shutdown_button = tk.Button(self.canvas, text="Sair", font=('Arial', 14), command=self.master.destroy, bg='#d8e6f4')
        self.canvas.create_window(780, 680, anchor='center', window=self.shutdown_button)

    def upload_img(self):
        try:
            # Pede o utilizador o endereço da imagem
            img_path = askopenfilename()
            if not img_path:  # Em caso do utilizador cancelar o endereçamento de imagem
                return None

            img = Image.open(img_path)
            img_width, img_height = img.size

            # Redimensiona a imagem para encaixar no canvas2 (400x600)
            if img_width > 400 or img_height > 600:
                while img_width > 400 or img_height > 600:
                    img_width *= .99
                    img_height *= .99
                img = img.resize((int(img_width), int(img_height)))

            # Converte a imagem para um formato que o Tkinter possa usar
            true_image = ImageTk.PhotoImage(img)

            #Põe a imagem desejada no local desejado
            self.canvas2.create_image(10, 10, anchor='nw', image=true_image)

            # Mantém a referência para evitar informções em desnecessárias e em excesso
            self.canvas2.image = true_image

            return img_path  # Retorna o endereçamento de imagem para uso futuro

        except UnidentifiedImageError:
            messagebox.showerror(title='Warning:', message='The image could not be read, please make sure the selected image is an image file.')

    def read_emotion(self):
        # pega o endereço de imagem do upload_img() 
        img_path = self.upload_img()
        if not img_path:  # If no image was uploaded, return early
            return

        # Lê a imagem usando o OpenCV
        image = cv2.imread(img_path)

        # Transfere a imagem para o DeepFace para a análise de emoções
        result = DeepFace.analyze(image, actions=("emotion",))

        # Se o resultado é uma lista (Pode acontecer se tiver mais de uma face), pega sempre o 1º elemento
        if isinstance(result, list):
            result = result[0]

        emotions = {key: round(float(value), 4) for key, value in result['emotion'].items()}
        print(f"Emotions: {emotions}")

        # Printa a emoção dominante
        print(f"Dominant emotion: {result['dominant_emotion']}")

        # Printa a confiança
        face_confidence = result.get('face_confidence', None) * 100
        print(f'Face confidence: {face_confidence}%')

        # Pega a emoção dominante e a confiança
        dominant_emotion = result['dominant_emotion']
        face_confidence = result.get('face_confidence', None) * 100

        # Mostra os resultados no Label
        result_text = f"Dominant Emotion: {dominant_emotion}\n"
        result_text += f"Confidence: {face_confidence}%\n"
        result_text += "Emotions: " + "\n".join([f"{key}: {value}" for key, value in emotions.items()])

        # Atualiza o emotion label no canvas
        self.emotion_label.config(text=result_text)

