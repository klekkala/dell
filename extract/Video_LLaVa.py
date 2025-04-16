from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from decord import VideoReader, cpu
import sys
import torch
import numpy as np
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")

device = "cuda:0"
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
tokenizer, model, image_processor, _ = load_pretrained_model(
    pretrained,
    model_base=None,
    model_name="llava_qwen",
    torch_dtype="float16",
    device_map={"": 0},
    attn_implementation="eager"
)

def load_video(video_path, max_frames_num, fps=1, force_sample=False, max_seconds=120):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frame_num / avg_fps

    max_frame_limit = int(min(total_frame_num, max_seconds * avg_fps))

    fps = round(avg_fps / fps)
    # frame_idx = [i for i in range(0, len(vr), fps)]
    frame_idx = [i for i in range(0, max_frame_limit, fps)]
    
    if len(frame_idx) > max_frames_num or force_sample:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i / avg_fps:.2f}s" for i in frame_idx])
    # return frames, frame_time, video_time
    return frames, frame_time, min(video_time, max_seconds)

video_path = sys.argv[1]
max_frames_num = 16
video_frames, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True, max_seconds=120)

video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].cuda().half()
video_tensor = [video_tensor]

conv_template = "qwen_1_5"
time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video_tensor[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video"
question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease describe this video in detail."
print(question)

conv = deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

torch.cuda.empty_cache()

# output = model.generate(
#     input_ids,
#     images=video_tensor,
#     modalities=["video"],
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=128,
#     use_cache=False
# )

with torch.no_grad():
    output = model(
        input_ids=input_ids,
        images=video_tensor,
        modalities=["video"]
    )
logits = output.logits
generated_ids = torch.argmax(logits, dim=-1)

text_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("Text_outputs:", text_outputs)