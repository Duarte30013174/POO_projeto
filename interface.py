from tkinter import Tk, Label, Frame, Button, Toplevel, Listbox, SINGLE, END, messagebox
from PIL import Image, ImageTk
import cv2
from poo import ImageManager  # Importando o restante do código

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.manager = ImageManager()
        self.panel = None
        self.create_widgets()

    def create_widgets(self):
        self.root.title("Gerenciamento de Imagens")
        self.root.geometry("650x650")

        self.panel = Label(self.root)
        self.panel.pack(pady=10)

        button_frame = Frame(self.root)
        button_frame.pack(pady=20)

        Button(button_frame, text="Imagem", command=self.open_image_selection, width=15).grid(row=0, column=0, padx=5)
        Button(button_frame, text="Restaurar", command=self.restore_image, width=15).grid(row=0, column=1, padx=5)
        Button(button_frame, text="Pixelizar", command=lambda: self.apply_effect("pixelate"), width=15).grid(row=0, column=2, padx=5)
        Button(button_frame, text="Adicionar Ruído", command=lambda: self.apply_effect("noise"), width=15).grid(row=0, column=3, padx=5)
        Button(button_frame, text="Avaliar Emoções", command=self.evaluate_emotions, width=15).grid(row=0, column=4, padx=5)

    def open_image_selection(self):
        selection_window = Toplevel(self.root)
        selection_window.title("Selecione uma imagem")
        selection_window.geometry("500x400")

        images = self.manager.list_all_images()
        if not images:
            messagebox.showerror("Erro", "Nenhuma imagem encontrada na pasta 'Pessoas'.")
            selection_window.destroy()
            return

        listbox = Listbox(selection_window, selectmode=SINGLE, height=15, width=60)
        for img in images:
            listbox.insert(END, img)
        listbox.pack(pady=10)

        def load_selected_image():
            selected_index = listbox.curselection()
            if not selected_index:
                messagebox.showerror("Erro", "Nenhuma imagem selecionada.")
                return

            img_path = listbox.get(selected_index)
            try:
                img = self.manager.load_image(img_path)
                self.display_image(img)
                selection_window.destroy()
            except ValueError as e:
                messagebox.showerror("Erro", str(e))

        Button(selection_window, text="Carregar", command=load_selected_image).pack(pady=10)

    def restore_image(self):
        img = self.manager.restore_image()
        self.display_image(img)

    def apply_effect(self, effect):
        img = self.manager.apply_effect(effect)
        self.display_image(img)

    def evaluate_emotions(self):
        try:
            # DeepFace análise de emoções
            emotions = self.manager.analyze_emotions()
            emotion_text = "\n".join([f"{emotion}: {value:.2f}%" for emotion, value in emotions.items()])

            # LBPH reconhecimento facial
            person_name, confidence = self.manager.recognize_face()
            lbph_text = f"Reconhecido como: {person_name} (Confiança: {confidence:.2f})"

            result_text = f"{lbph_text}\n\nEmoções Detectadas:\n{emotion_text}"
            messagebox.showinfo("Resultados", result_text)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao avaliar: {e}")

    def display_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((375, 500), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.panel.config(image=img_tk)
        self.panel.image = img_tk

if __name__ == "__main__":
    root = Tk()
    app = ImageApp(root)
    root.mainloop()