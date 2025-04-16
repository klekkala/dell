# extract_video_game

This sub project is mainly used to extract image frames from game videos, save them to a designated folder, and then extract the reward value through the foundation model, as well as describe the caption of the corresponding frame in certain intervals

## instruction

### 1. Create env

If you need to use the corresponding large model for testing, create your own ".env" file and fill in OPENAI-API_KEY, GOOGLE-API_KEY, and LLAMA_KEY.

### 2. Download video game

Download game videos that need to be processed from YouTube

```bash
python download_video.py "<video_url>" "<output_path>"
```

### 3. Extract frame from video file

```bash
python extract_video_frame.py "<video_path>"
```

### 4. Extract reward score number from frame

You can choose different LLMs to help extract reward numbers for specific areas in the video frame. Note that you need to set the corresponding bounding box in the code to capture and identify the target area. This is hard coded now and usually uses the method of reading CSV files. The results obtained from different models are unstable and require manual inspection.


```bash
python extract_frame_reward_gemini.py "<video_frame_dir>"

python extract_frame_reward_openai.py "<video_frame_dir>"

python extract_frame_reward_llama.py "<video_frame_dir>"
```

## 5. Generate text using image description models (similar to narration)

Generate narration based on extracted frames at certain intervals. Specific settings require modifying the code.

The obtained format is similar to "frame_000029.png: the player is xxxx". In the future, the format that can be modified to a timestamp format, similar to "00:01: the player is xxxx".


```bash
python extract_frame_caption_gemini.py "<video_frame_dir>"
```

## 6. (Experimental) Describe the video through LLaVA Video

Using lmms-lab/LLaVA-Video-7B-Qwen2 on Huggingface to describe videos and generate more coherent narratives. Higher GPU requirements need to be noted before running. The current running results are not very good, and further debugging is needed.

```bash
python Video_LLaVa.py "<video_path>"
```



