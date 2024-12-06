import cv2
import os
from deepface import DeepFace
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Não foi possível carregar {image_path}")
    
    def convert_to_gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def save_image(self, output_path):
        cv2.imwrite(output_path, self.image)

class FaceDetector(ImageProcessor):
    def __init__(self, image_path, output_folder, person_name="Unknown"):
        super().__init__(image_path)
        self.output_folder = output_folder
        self.person_name = person_name
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self):
        gray = self.convert_to_gray()
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        output_files = []

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                face = self.image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200))
                output_path = os.path.join(self.output_folder, f'clipped_face_{self.person_name}_{i}.jpg')
                self.save_image(output_path)
                output_files.append(output_path)
                print(f"Rosto {i + 1} salvo como '{output_path}'")
        else:
            print(f"Nenhum rosto detectado em {self.image_path}")
        
        return output_files

class EmotionAnalyzer(ImageProcessor):
    def analyze_emotions(self):
        try:
            result = DeepFace.analyze(img_path=self.image_path, actions=['emotion'])
            return result[0]['emotion']
        except Exception as e:
            print(f"Erro ao analisar emoções para {self.image_path}: {e}")
            return {}

class ImageEffect(ImageProcessor):
    def pixelate(self, block_size=10):
        (h, w) = self.image.shape[:2]
        temp = cv2.resize(self.image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated
    
    def add_noise(self, noise_level=0.05):
        noise = np.random.normal(0, 255 * noise_level, self.image.shape).astype(np.uint8)
        noisy_image = cv2.add(self.image, noise)
        return noisy_image
