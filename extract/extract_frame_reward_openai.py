import os
import sys
from PIL import Image
import base64
import openai
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

image_folder = sys.argv[1]   
output_folder = sys.argv[1] 
output_file = os.path.join(output_folder, "mappings.txt")

# Load bounding box
# Normally, the corresponding columns can be read from the CSV file
bounding_box = (235, 10, 385, 35)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

with open(output_file, "w") as f:
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.endswith(".png"):
            image_path = os.path.join(image_folder, image_name)
            
            image = Image.open(image_path)
            cropped_image = image.crop(bounding_box)
            temp_image_path = "temp_cropped.png"
            cropped_image.save(temp_image_path)

            image_base64 = encode_image_to_base64(temp_image_path)
            
            messages = [
                {"role": "system", "content": "You are an assistant that recognizes numbers from cropped images. You are given Base64 encoded images."},
                {
                    "role": "user",
                    "content": f"This is a Base64 encoded image:\n{image_base64}\n\nPlease analyze the image and return the detected a number in the image. If you can not find any number, return 0. Only return a number."
                }
            ]
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )

                print(response)

                recognized_text = response["choices"][0]["message"]["content"].strip()
                
                if not recognized_text.isdigit():
                    recognized_text = "0"
                
            except Exception as e:
                print(f"Detecting picture {image_name} error: {e}")
                recognized_text = "0"
            
            f.write(f"{image_name}, {recognized_text}\n")
            print(f"Processed: {image_name}, Result: {recognized_text}")

print(f"All the pictures finish process, result save to {output_file}")