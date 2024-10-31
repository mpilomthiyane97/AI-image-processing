import gradio as gr
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
from gtts import gTTS

# Load the pre-trained BLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def opencv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def text_to_speech(text, frame_number=None):
    tts = gTTS(text, lang='en')
    audio_path = f"frame_{frame_number}_audio.mp3" if frame_number is not None else "audio.mp3"
    tts.save(audio_path)
    return audio_path

def process_video(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    audio_files = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            caption = generate_caption(opencv_to_pil(frame))
            audio_file = text_to_speech(caption, frame_count)
            audio_files.append(audio_file)

        frame_count += 1

    cap.release()
    return audio_files

def handle_input(image=None, video=None):
    if image:
        # Process image
        caption = generate_caption(image)
        audio_path = text_to_speech(caption)
        return caption, audio_path
    elif video:
        # Process video
        audio_files = process_video(video)
        return "Video processed. Audio files generated.", audio_files
    else:
        return "No input provided", None

# Define the Gradio interface
interface = gr.Interface(
    fn=handle_input,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        # gr.Video(label="Upload a Video")
    ],
    outputs=[
        gr.Textbox(label="Description"),
        gr.Audio(label="Audio Description")
    ],
    title="Image and Video Captioning with Audio Conversion",
    description="Upload an image or video to generate a caption and audio description. For videos, audio descriptions are generated for each frame."
)

# Launch the Gradio interface
interface.launch()
