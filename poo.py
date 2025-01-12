import cv2
import os
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from deepface import DeepFace

# Caminho raiz onde estão as imagens
ROOT_FOLDER = "Pessoas"

class ImageManager:
    def __init__(self):
        self.img_path = None
        self.original_image = None
        self.current_image = None

    def list_all_images(self):
        """Lista todas as imagens na pasta ROOT_FOLDER."""
        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = []
        for root, _, files in os.walk(ROOT_FOLDER):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files

    def load_image(self, img_path):
        """Carrega a imagem selecionada e a define como imagem atual."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Erro ao carregar a imagem.")
        self.img_path = img_path
        self.original_image = img.copy()
        self.current_image = img.copy()
        return img

    def restore_image(self):
        """Restaura a imagem original."""
        self.current_image = self.original_image.copy()
        return self.current_image

    def apply_effect(self, effect):
        """Aplica um efeito à imagem atual."""
        if effect == "pixelate":
            h, w = self.current_image.shape[:2]
            temp = cv2.resize(self.current_image, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
            self.current_image = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        elif effect == "noise":
            noise = np.random.normal(0, 25, self.current_image.shape).astype(np.uint8)
            self.current_image = cv2.add(self.current_image, noise)
        return self.current_image

    def analyze_emotions(self):
        """Avalia emoções da imagem atual usando DeepFace."""
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, self.current_image)
        try:
            result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']
            return emotions
        finally:
            os.remove(temp_path)

    def recognize_face(self):
        """Reconhece o rosto usando LBPH (Local Binary Patterns Histograms)."""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        images, labels = self.prepare_training_data()
        recognizer.train(images, np.array(labels))

        if self.current_image is None:
            raise ValueError("Nenhuma imagem carregada para reconhecimento.")

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        label, confidence = recognizer.predict(gray_image)

        person_name = self.get_person_name_from_label(label)
        return person_name, confidence

    def prepare_training_data(self):
        """Prepara os dados de treino para o LBPH usando as imagens na pasta 'Pessoas'."""
        images = []
        labels = []
        label_map = {}
        current_label = 0

        for person_folder in os.listdir(ROOT_FOLDER):
            person_path = os.path.join(ROOT_FOLDER, person_folder)
            if os.path.isdir(person_path):
                label_map[current_label] = person_folder
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    images.append(image)
                    labels.append(current_label)
                current_label += 1

        self.label_map = label_map
        return images, labels

    def get_person_name_from_label(self, label):
        """Obtém o nome da pessoa a partir do rótulo."""
        return self.label_map.get(label, "Desconhecido")