import cv2
import os
import random
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from deepface import DeepFace
from PIL import Image, ImageTk

# Função para carregar e mostrar imagem
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        global img_path, original_image
        img_path = file_path
        img = cv2.imread(img_path)
        original_image = img.copy()  # Salva a imagem original para restaurar depois
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB para o tkinter exibir corretamente
        img = cv2.resize(img, (400, 400))  # Redimensionar a imagem para o tamanho exibido
        img = ImageTk.PhotoImage(Image.fromarray(img))
        panel.config(image=img)
        panel.image = img

# Função para restaurar a imagem ao seu estado original
def restore_image():
    if not img_path or original_image is None:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))  # Redimensionar para o tamanho exibido
    img = ImageTk.PhotoImage(Image.fromarray(img))

    panel.config(image=img)
    panel.image = img

# Função para aplicar DeepFace e mostrar emoções
def recognize_emotions():
    if not img_path:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'])
        result = result[0]  # Pega o primeiro item da lista

        # Exibe as emoções e suas porcentagens
        emotions_text = "Emoções detectadas:\n"
        total_score = sum(result['emotion'].values())
        for emotion, score in result['emotion'].items():
            percentage = (score / total_score) * 100
            emotions_text += f"{emotion.capitalize()}: {percentage:.2f}%\n"
        
        emotion_label.config(text=emotions_text)

    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao analisar a imagem: {str(e)}")

# Função para pixelizar a imagem
def pixelate_image():
    if not img_path:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Diminuir o tamanho da imagem (para efeito de pixelização)
    small = cv2.resize(img, (width // 10, height // 10), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('pixelated_image.jpg', pixelated)
    img = cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))  # Redimensionar para o tamanho menor
    img = ImageTk.PhotoImage(Image.fromarray(img))

    panel.config(image=img)
    panel.image = img

# Função para adicionar ruído aleatório à imagem
def add_random_noise():
    if not img_path:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    img = cv2.imread(img_path)
    row, col, ch = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.array(img, dtype=float) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv2.imwrite('noisy_image.jpg', noisy)
    img = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))  # Redimensionar para o tamanho menor
    img = ImageTk.PhotoImage(Image.fromarray(img))

    panel.config(image=img)
    panel.image = img

# Configuração da interface Tkinter
root = Tk()
root.title("Reconhecimento de Emoções")
root.geometry("600x700")  # Dimensão da janela

img_path = None
original_image = None  # Variável para armazenar a imagem original

# Painel para exibir a imagem
panel = Label(root)
panel.pack(pady=10)

# Container para os botões, alinhando-os no centro
button_frame = Frame(root)
button_frame.pack(pady=20)

# Botão para carregar imagem
browse_button = Button(button_frame, text="Pesquisar", command=browse_image, width=12)
browse_button.grid(row=0, column=0, padx=10)

# Botão para reconhecer emoções
recognize_button = Button(button_frame, text="Reconhecer", command=recognize_emotions, width=12)
recognize_button.grid(row=0, column=1, padx=10)

# Botão para restaurar imagem ao normal
restore_button = Button(button_frame, text="Normal", command=restore_image, width=12)
restore_button.grid(row=0, column=2, padx=10)

# Botão para pixelizar imagem
pixelate_button = Button(button_frame, text="Pixelizar", command=pixelate_image, width=12)
pixelate_button.grid(row=0, column=3, padx=10)

# Botão para adicionar ruído aleatório
random_noise_button = Button(button_frame, text="Ruído", command=add_random_noise, width=12)
random_noise_button.grid(row=0, column=4, padx=10)

# Rótulo para exibir emoções
emotion_label = Label(root, text="Emoções detectadas aqui", justify=LEFT, wraplength=500)
emotion_label.pack(pady=10)

# Executar a interface Tkinter
root.mainloop()

