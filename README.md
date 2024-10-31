# Overview
In the previous project, you processed a video by dividing it into multiple frames then passing them into the model to get the description. Now we will do a simple twist to the previous code by giving your code the power of speech.

# Models
- You will use the following models:
  - Salesforce/blip-image-captioning-base model: A pre-trained model for generating captions for images which can be found [here](https://huggingface.co/Salesforce/blip-image-captioning-base).
  - Microsoft/SpeechT5: A pre-trained model for generating audio from text which can be found [here](https://huggingface.co/microsoft/speecht5_tts)
# Instructions
Your code should be capable of doing the following:
1. Accept a single image. We're going back to the roots with this one. So we will be accepting images in the beginning.
2. The images are passed into the blip model to explain its content.
3. The description, instead of being passed as simple text to Gradio, will be converted into audio the same way you studied in class today.
4. Once the audio has been generated, pass the audio to Gradio and listen to the sound.
5. (Bonus) if this was to easy for you, go back to the previous project and move the code you made the processes the video in a frame by frame manner. Pass the description to generate audio.

