import os
import sys
import llama_ocr
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLAMA_KEY")

def extract_text_from_image(image_path, api_key):
    result = llama_ocr.ocr(image_path, api_key=api_key)
    return result

image_folder = sys.argv[1]  
output_folder = sys.argv[1]  
output_file = os.path.join(output_folder, "mapping.txt")

# Load bounding box
# Normally, the corresponding columns can be read from the CSV file
bounding_box = (235, 10, 385, 35)

with open(output_file, "w") as f:
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.endswith(".png"):
            image_path = os.path.join(image_folder, image_name)
            
            try:
                api_key = api_key
                extracted_text = extract_text_from_image(image_path, api_key)
                print(f"Picture {image_name} detect result:{extracted_text}")
                
                if not extracted_text.isdigit():
                    extracted_text = "0"
                
            except Exception as e:
                print(f"Detect picture {image_name} error: {e}")
                extracted_text = "0"
            
            f.write(f"{image_name}, {extracted_text}\n")
            print(f"Process: {image_name}, result: {extracted_text}")

            break

print(f"All the pictures finish process, result save to {output_file}")