import cv2
import os
from deepface import DeepFace
import numpy as np

def detect_faces(image_path, output_folder, person_name="Unknown"):
    """
    Detecta e recorta rostos de uma imagem.
    Args:
        image_path (str): Caminho da imagem de entrada.
        output_folder (str): Caminho para salvar os rostos recortados.
        person_name (str): Nome da pessoa (opcional, usado para nomear os arquivos).
    Returns:
        list: Lista de caminhos dos arquivos de rostos recortados.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Não foi possível carregar {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    output_files = []
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))
            output_path = os.path.join(output_folder, f'clipped_face_{person_name}_{i}.jpg')
            cv2.imwrite(output_path, face_resized)
            output_files.append(output_path)
            print(f"Rosto {i + 1} salvo como '{output_path}'")
    else:
        print(f"Nenhum rosto detectado em {image_path}")
    
    return output_files

def analyze_emotions(image_path):
    """
    Analisa as emoções de um rosto usando DeepFace.
    Args:
        image_path (str): Caminho para a imagem do rosto.
    Returns:
        dict: Resultados da análise de emoções.
    """
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        return result[0]['emotion']  # Retorna apenas as emoções
    except Exception as e:
        print(f"Erro ao analisar emoções para {image_path}: {e}")
        return {}

def pixelate_image(image_path, block_size=10):
    """
    Aplica um efeito de pixelização a uma imagem.
    Args:
        image_path (str): Caminho da imagem.
        block_size (int): Tamanho dos blocos para pixelização.
    Returns:
        np.array: Imagem pixelada.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Não foi possível carregar {image_path}")
        return None

    (h, w) = image.shape[:2]
    temp = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def add_random_noise(image_path, noise_level=0.05):
    """
    Adiciona ruído aleatório a uma imagem.
    Args:
        image_path (str): Caminho da imagem.
        noise_level (float): Intensidade do ruído.
    Returns:
        np.array: Imagem com ruído.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Não foi possível carregar {image_path}")
        return None

    noise = np.random.normal(0, 255 * noise_level, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image