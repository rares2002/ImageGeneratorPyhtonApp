import keras_cv
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import numpy as np
import os
import pyttsx3
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip

image_to_description_dictionary = {
    "image_v2_1.png": "A realistic image of a fuchsia-colored carrot",
    "image_v2_1_extracted.png": "The extracted image of the carrot",
    "image_v2_1_extracted_yuv.png": "The YUV color space representation of the extracted carrot",
    "image_v2_1_mask.png": "The binary mask of the carrot",
    "image_v2_1_rectangle.png": "The original image with a red rectangle around the carrot",
    "image_v2_1_yuv.png": "The YUV color space representation of the original image",

}


def generate_image():
    model = keras_cv.models.StableDiffusion(
        img_width=512, img_height=512, jit_compile=False
    )
    return model.text_to_image("Generate a realistic image of a fuchsia-colored carrot in a"
                               "meadow under a clear blue sky.", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


def save_images(images):
    for i in range(len(images)):
        img = Image.fromarray(images[i])
        img.save(f"image_v2_{i}.png")


def draw_rectangle():
    im = Image.open('./images/image_v2_1.png')
    fig, ax = plt.subplots()
    ax.imshow(im)

    rect = patches.Rectangle((160, 60), 210, 400, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.axis('off')
    #set dimension to 512x512 pixels
    fig.set_size_inches(5.12, 5.12)
    fig.savefig('./images/image_v2_1_rectangle.png', bbox_inches='tight', pad_inches=0)
    resized = cv2.resize(cv2.imread('./images/image_v2_1_rectangle.png'), (512, 512))
    cv2.imwrite('./images/image_v2_1_rectangle.png', resized)


def extract_object():
    img = cv2.imread('./images/image_v2_1.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([150, 100, 100])
    upper_range = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite('./images/image_v2_1_extracted.png', result)
    cv2.imwrite('./images/image_v2_1_mask.png', mask)


def convert_to_yuv():
    original_img = cv2.imread('./images/image_v2_1.png')
    extracted_img = cv2.imread('./images/image_v2_1_extracted.png')
    original_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
    extracted_yuv = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2YUV)
    cv2.imwrite('./images/image_v2_1_yuv.png', original_yuv)
    cv2.imwrite('./images/image_v2_1_extracted_yuv.png', extracted_yuv)


def generate_audio():
    image_folder = './images'
    audio_folder = './audio'
    os.makedirs(audio_folder, exist_ok=True)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    audio_durations = []

    for image in images:
        text = image_to_description_dictionary[image]
        filename = image[:-4]
        audio_file = os.path.join(audio_folder, f"{filename}.wav")
        engine.save_to_file(text, audio_file)
        engine.runAndWait()

        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000  # Durata este în secunde
        audio_durations.append(duration)

    return audio_durations


def create_video(audio_durations):
    image_folder = './images'
    audio_folder = './audio'
    video_folder = './videos'

    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order

    audios = [os.path.join(audio_folder, audio) for audio in os.listdir(audio_folder) if audio.endswith(".wav")]
    audios.sort()  # Ensure the audios are in the correct order
    print(images)
    print(audios)
    # Create a list to store the video clips
    video_clips = []
    print(images)
    for i, img in enumerate(images):
        # Create a video clip from the image
        print(audio_durations[i])
        img_array = cv2.imread(img)
        img_converted = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        img_array_with_text = cv2.putText(img_converted, image_to_description_dictionary[os.path.basename(img)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        img_clip = ImageSequenceClip([img_array_with_text], durations=[audio_durations[i]])

        # Create a text clip for the image
        text = image_to_description_dictionary[os.path.basename(img)]
        # text_clip = TextClip(text, fontsize=24, color='white').set_duration(audio_durations[i])
        #
        # # Overlay the text clip on the image clip
        video_clip = CompositeVideoClip([img_clip])

        video_clips.append(video_clip)
    # Concatenate all video clips
    final_clip = concatenate_videoclips(video_clips)

    # Create a list to store the audio clips
    audio_clips = [AudioFileClip(audio) for audio in audios]

    # Concatenate all audio clips
    audio = concatenate_audioclips(audio_clips)

    # Set the audio of the clip
    final_clip.audio = audio

    # Write the result to a file
    final_clip.write_videofile(os.path.join(video_folder, "final_video.mp4"), codec='mpeg4', fps=24, audio_codec='aac')


def main():
    # images = generate_image()
    # plot_images(images)
    # save_images(images)
    draw_rectangle()
    # extract_object()
    # convert_to_yuv()
    audio_durations = generate_audio()
    print(audio_durations)
    create_video(audio_durations)


main()
