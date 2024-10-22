import gradio as gr
import cv2
import torch
from ultralytics import YOLO
import time

# Miramos si CUDA está disponible 
print(torch.cuda.is_available())

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('bestDroneDetection.pt')  # Puedes usar otro modelo como yolov8s.pt, yolov8m.pt, etc.


# Función para procesar el video y actualizar la interfaz frame a frame en tiempo real
def video_frame_update(video):
    cap = cv2.VideoCapture(video)
    
    # Verificar si el archivo de video se abrió correctamente
    if not cap.isOpened():
        yield gr.update(visible=False), gr.update(value="Error: no se pudo abrir el archivo de video.")
        return
    
    # Para mostrar la velocidad de inferencia en FPS
    prev_time = time.time()

    # Procesar el video frame a frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Iniciar el temporizador para medir el tiempo de inferencia
        start_time = time.time()

        # Convertimos la imagen a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Realizar la detección de objetos en el frame actual
        results = model(frame)

        # Dibujar los cuadros delimitadores en el frame
        annotated_frame = results[0].plot()

        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - start_time)

        # Sub-imagen de "person" detectada
        drone_subimage = None

        # Buscar si hay alguna detección de la clase "person"
        for result in results[0].boxes:
            cls_id = int(result.cls[0])
            if model.names[cls_id] == "drone":
                # Obtener las coordenadas del bounding box
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                # Recortar la sub-imagen de la persona detectada
                drone_subimage = frame[y1:y2, x1:x2]
                # Get the original dimensions
                (h, w) = drone_subimage.shape[:2]

                # Desired width
                new_height = 200

                # Calculate the aspect ratio
                aspect_ratio = w / h
                new_width = int(new_height * aspect_ratio)

                # Resize the image
                resized_image = cv2.resize(drone_subimage, (new_width, new_height))
                #break  # Solo necesitamos una detección de "drone"

        # Simular el tiempo de procesamiento para cada frame (si es necesario)
        # time.sleep(0.05)  # Simulación de tiempo de procesamiento entre frames

        # Mostrar el frame anotado en tiempo real, el FPS y la sub-imagen (si se detectó una persona)
        yield gr.update(value=annotated_frame), gr.update(value=f"{fps:.2f} FPS"), gr.update(value=resized_image if drone_subimage is not None else None)

    cap.release()

# Crear la interfaz en Gradio
with gr.Blocks() as demo:
    gr.Markdown("# AI4HyDrop - Demo of Drone Detection Service")
    gr.Image("AI4HyDrop_logo.png", show_download_button=False, container=False, show_fullscreen_button=False, width=400)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Video source (open file or use external camera)")
            fps_text = gr.Textbox(label="Inference speed (FPS)", interactive=False)
        with gr.Column(scale=2):
            video_output = gr.Image(label="Drone Detection", visible=True, type="numpy")
        with gr.Column(scale=1):            
            person_image = gr.Image(label="Detected oObject", type="numpy")

    # Mensaje de carga que se activará mientras se procesa el video
    loading_text = gr.Textbox(value="Loading video and detecting objects, please wait...", visible=False)
    
    # Configurar los eventos
    video_input.change(
        video_frame_update, 
        inputs=video_input, 
        outputs=[video_output, fps_text, person_image],
    )

# Habilitar la cola para usar funciones generadoras
demo.queue()

# Ejecutar la interfaz
demo.launch()
