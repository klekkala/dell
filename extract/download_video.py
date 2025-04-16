# from pytube import YouTube
from pytubefix import YouTube
import sys


def download_video(video_url, output_path=None):
    try:
        # Create a YouTube object
        yt = YouTube(video_url, use_po_token=True)

        stream = yt.streams.get_highest_resolution()
        # stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

        # download the video stream, higher resolution
        # stream = yt.streams.filter(adaptive=True, only_video=True, file_extension='mp4').order_by('resolution').desc().first()

        # If output path is not provided, use default filename
        if output_path is None:
            output_path = stream.default_filename

        # Download the video
        stream.download(output_path=output_path)

        print("Download completed!")
    except Exception as e:
        print("An error occurred:", str(e))

def download_video_name(video_url, video_name, output_path=None):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest resolution stream
        stream = yt.streams.get_highest_resolution()

        # If output path is not provided, use default filename
        if output_path is None:
            output_path = stream.default_filename

        # Download the video
        stream.download(output_path=output_path, filename=video_name)

        print("Download completed!")
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage
if __name__ == "__main__":
    video_url = sys.argv[1]
    output_path = sys.argv[2]
    download_video(video_url, output_path)