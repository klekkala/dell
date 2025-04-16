import os
import sys
import time
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

image_folder = sys.argv[1]
output_file = os.path.join(image_folder, "captions.txt")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

FRAME_STEP = 30

caption_prompt = "This is a frame from a first-person game video. Describe what the player is doing or what is happening in this scene in one sentence."

with open(output_file, "a", encoding="utf-8") as f:
    for i, image_name in enumerate(sorted(os.listdir(image_folder))):
        if not image_name.endswith(".png"):
            continue

        if i % FRAME_STEP != 0:
            continue

        image_path = os.path.join(image_folder, image_name)
        print(f"Processing: {image_name}")

        try:
            # upload image to Gemini
            image = Image.open(image_path)
            sample_file = genai.upload_file(path=image_path, display_name=image_name)

            response = model.generate_content([caption_prompt, sample_file])
            caption = response.candidates[0].content.parts[0].text.strip()

            # write down return
            f.write(f"{image_name}: {caption}\n")
            print(f"{image_name}: {caption}")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            f.write(f"{image_name}: [ERROR]\n")

        time.sleep(2)

print(f"Captioning complete. Results saved to {output_file}")