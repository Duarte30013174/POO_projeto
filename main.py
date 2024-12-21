import os
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from teste import ImageProcessor, EmotionAnalyzer, ImageEffect

ROOT_FOLDER = "Pessoas"
img_path = None
original_image = None
current_image = None  # Imagem atualmente exibida

# Listar imagens na pasta
def list_all_images():
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = []
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

# Seleção de imagem
def open_image_selection():
    global img_path, original_image, current_image

    selection_window = Toplevel(root)
    selection_window.title("Selecione uma imagem")
    selection_window.geometry("500x400")

    images = list_all_images()
    if not images:
        messagebox.showerror("Erro", "Nenhuma imagem encontrada na pasta 'Pessoas'.")
        selection_window.destroy()
        return

    listbox = Listbox(selection_window, selectmode=SINGLE, height=15, width=60)
    for img in images:
        listbox.insert(END, img)
    listbox.pack(pady=10)

    def load_selected_image():
        global img_path, original_image, current_image
        selected_index = listbox.curselection()
        if not selected_index:
            messagebox.showerror("Erro", "Nenhuma imagem selecionada.")
            return

        img_path = listbox.get(selected_index)
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Erro", "Erro ao carregar a imagem.")
            return

        original_image = img.copy()
        current_image = img.copy()
        display_image(img)
        selection_window.destroy()

    load_button = Button(selection_window, text="Carregar", command=load_selected_image)
    load_button.pack(pady=10)

def restore_image():
    global original_image, current_image
    if original_image is None:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    current_image = original_image.copy()
    display_image(original_image)

def pixelate_image():
    global current_image
    if current_image is None:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    try:
        processor = ImageEffect(image_path=None)  # Caminho não necessário para operações no array
        processor.image = current_image  # Define a imagem atual
        current_image = processor.pixelate(pixel_size=10)  # Aplica pixelização
        display_image(current_image)
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao pixelizar a imagem: {e}")

def add_noise_image():
    global current_image
    if current_image is None:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    try:
        processor = ImageEffect(image_path=None)  # Caminho não necessário para operações no array
        processor.image = current_image  # Define a imagem atual
        current_image = processor.add_noise(noise_level=0.7)  # Adiciona ruído
        display_image(current_image)
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao adicionar ruído: {e}")


def evaluate_emotions():
    global current_image
    if current_image is None:
        messagebox.showerror("Erro", "Nenhuma imagem carregada.")
        return

    try:
        # Analisar emoções usando o EmotionAnalyzer
        processor = EmotionAnalyzer(image_path=None)  # Caminho não necessário, passaremos o array
        emotions = processor.analyze_emotions(image_array=current_image)
        result_text = "\n".join([f"{emotion}: {value:.2f}%" for emotion, value in emotions.items()])
        messagebox.showinfo("Emoções Detectadas", result_text)
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao avaliar emoções: {e}")



def display_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((375, 500), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk



root = Tk()
root.title("Gerenciamento de Imagens")
root.geometry("650x650")

panel = Label(root)
panel.pack(pady=10)

button_frame = Frame(root)
button_frame.pack(pady=20)

list_button = Button(button_frame, text="Imagem", command=open_image_selection, width=15)
list_button.grid(row=0, column=0, padx=5)

restore_button = Button(button_frame, text="Restaurar", command=restore_image, width=15)
restore_button.grid(row=0, column=1, padx=5)

pixelate_button = Button(button_frame, text="Pixelizar", command=pixelate_image, width=15)
pixelate_button.grid(row=0, column=2, padx=5)

noise_button = Button(button_frame, text="Adicionar Ruído", command=add_noise_image, width=15)
noise_button.grid(row=0, column=3, padx=5)

emotion_button = Button(button_frame, text="Avaliar Emoções", command=evaluate_emotions, width=15)
emotion_button.grid(row=0, column=4, padx=5)

root.mainloop()