import cv2
import os
from deepface import DeepFace
import numpy as np

class ImageProcessor:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.image = cv2.imread(image_path) if image_path else None
        if image_path and self.image is None:
            raise ValueError(f"Não foi possível carregar {image_path}")
    
    def set_image(self, image_array):
        """Define a imagem a partir de um array do OpenCV."""
        self.image = image_array

    def convert_to_gray(self):
        """Converte a imagem para escala de cinza."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def save_image(self, output_path):
        """Salva a imagem no caminho especificado."""
        cv2.imwrite(output_path, self.image)



class FaceDetector(ImageProcessor):
    def __init__(self, image_path=None, output_folder=None, person_name="Unknown"):
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
            print(f"Nenhum rosto detectado na imagem.")
        
        return output_files



class EmotionAnalyzer(ImageProcessor):
    def analyze_emotions(self, image_array=None):
        try:
            # Se uma imagem em array for passada, salvar temporariamente
            if image_array is not None:
                temp_path = "temp_image.jpg"
                cv2.imwrite(temp_path, image_array)
                result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
                os.remove(temp_path)  # Remover a imagem temporária
            else:
                # Caso contrário, usar o caminho da imagem
                result = DeepFace.analyze(img_path=self.image_path, actions=['emotion'], enforce_detection=False)
            return result[0]['emotion']
        except Exception as e:
            print(f"Erro ao analisar emoções: {e}")
            return {}



class ImageEffect(ImageProcessor):
    def pixelate(self, pixel_size=10):
        """Aplica pixelização à imagem."""
        return cv2.resize(
            cv2.resize(self.image, (self.image.shape[1] // pixel_size, self.image.shape[0] // pixel_size)),
            (self.image.shape[1], self.image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    def add_noise(self, noise_level=0.05):
        noise = np.random.normal(0, 255 * noise_level, self.image.shape).astype(np.uint8)
        noisy_image = cv2.add(self.image, noise)
        return noisy_image


