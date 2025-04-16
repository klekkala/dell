import os
import sys
import time
import base64
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

image_folder = sys.argv[1] 
output_folder = sys.argv[1]  
output_file = os.path.join(output_folder, "mapping.txt")

# Load bounding box
# Normally, the corresponding columns can be read from the CSV file
bounding_box = (1450, 40, 220, 60)

# Process bounding box data in different formats and modify the code according to specific situations
x, y, w, h = bounding_box
left = x
top = y
right = x + w
bottom = y + h
pil_bounding_box = (left, top, right, bottom)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

start_index = 000
start_image = f"frame_{start_index:06d}.png"

with open(output_file, "a") as f:
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.endswith(".png") and image_name >= start_image:
            image_path = os.path.join(image_folder, image_name)
            
            image = Image.open(image_path)
            cropped_image = image.crop(pil_bounding_box)
            temp_image_path = "temp_cropped.png"
            cropped_image.save(temp_image_path)

            prompt = "This is a cropped image from an Game frame. Please detect and return the number visible in the image. If no number is detected, return 0."

            try:
                sample_file = genai.upload_file(path=temp_image_path, display_name="Game frame")
                print(f"File uploaded successfully: {sample_file}")
            except Exception as e:
                print(f"Error uploading file: {e}")

            prompt_with_image = [prompt, sample_file]

            try:
                response = model.generate_content(prompt_with_image)
                print(response)

                recognized_text = response.candidates[0].content.parts[0].text.strip()

                if not recognized_text.replace('.', '', 1).isdigit():
                    recognized_text = "0"

            except Exception as e:
                print(f"Detecting picture {image_name} error: {e}")
                recognized_text = "0"

            f.write(f"{image_name}, {recognized_text}\n")
            print(f"Processed: {image_name}, result: {recognized_text}")

            time.sleep(2)


print(f"All the pictures finish process, result save to {output_file}")