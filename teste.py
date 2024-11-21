import cv2
import os
from deepface import DeepFace

# Cria uma pasta para salvar os rostos recortados
output_folder = 'Clipped_Faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carrega o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Diretório base onde as pastas das pessoas estão localizadas
base_dir = 'Pessoas'

# Verifica se o diretório existe
if not os.path.exists(base_dir):
    print(f"O diretório {base_dir} não foi encontrado.")
    exit(1)

# Percorre as pastas dentro de 'Pessoas'
for person_name in os.listdir(base_dir):
    person_folder = os.path.join(base_dir, person_name)

    # Verifica se é realmente uma pasta
    if not os.path.isdir(person_folder):
        continue

    # Percorre as imagens dentro da pasta de cada pessoa
    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)

        # Processa apenas arquivos de imagem (jpg, jpeg, png)
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Carrega a imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"Não foi possível carregar {image_path}")
                continue

            # Converte a imagem para tons de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detecta rostos na imagem
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Verifica se rostos foram detectados
            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    # Recorta cada rosto da imagem
                    face = image[y:y + h, x:x + w]

                    # Redimensiona o rosto para um tamanho padrão (200x200 pixels)
                    face_resized = cv2.resize(face, (200, 200))

                    # Define o caminho de saída e salva o rosto recortado
                    output_path = os.path.join(output_folder, f'clipped_face_{person_name}_{i}.jpg')
                    cv2.imwrite(output_path, face_resized)
                    print(f"\nRosto {i + 1} de {person_name} recortado e salvo como '{output_path}'")

                    # Aplicar a análise de emoção com DeepFace
                    result = DeepFace.analyze(img_path=output_path, actions=['emotion'])

                    # Aqui, verificamos se 'result' é uma lista e acessamos o primeiro item
                    result = result[0]  # Acessa o primeiro item da lista

                    # Processa e exibe as emoções com porcentagens
                    print(f"Emoções detectadas em {person_name}:")
                    total_score = sum(result['emotion'].values())  # Soma os valores das emoções
                    for emotion, score in result['emotion'].items():
                        percentage = (score / total_score) * 100
                        print(f"  {emotion.capitalize()}, Percentagem: {percentage:.2f}%")
            else:
                print(f"Nenhum rosto encontrado em {image_path}")

# Fecha todas as janelas do OpenCV (se houver alguma aberta)
cv2.destroyAllWindows() 