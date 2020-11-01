import sys
import subprocess
import pkg_resources
from os import listdir
from os.path import isdir
import imageio
import re
import datetime
import time
import json
import numpy
import matplotlib.pyplot as plt

##################################################
# Global variables to specify where stuff goes

captions_folder = "captions"
videos_folder = "videos"
frames_folder = "frames"
frame_rate_default = 5

##################################################
# Captions-related methods

def download_captions(yt_video):
    """
    Downloads captions from a youtube video, stored in a vtt file

    Parameters:
    - yt_video: the link to the youtube video to get captions for.
    If this link doesn't contain the text "youtube.com", this method will throw an exception.
    Also, anything that the youtube_dl package doesn't link will cause abnormal behavior.
    Example of a valid link: "https://www.youtube.com/watch?v=Wn4Kxy6bPI4"

    REQUIRES: youtube-dl

    NOTE: be sure to run the `install_youtubedl()` method before 
    making the first call to this method
    to ensure the user has youtube-dl
    """
    
    checkValidYoutubeVideo(yt_video)
    makeFolderIfNotExists(captions_folder)

    dl_captions_command = ["youtube-dl", "--output", f"{captions_folder}/%(title)s-%(id)s.%(ext)s", "--all-subs", "--skip-download", yt_video]
    subprocess.check_call(dl_captions_command)

def get_caption_list(yt_video, language="en"):
    """
    Gets the captions to display for a given time

    Parameters:
    - yt_video: the link to the youtube video to get captions for. 
    If this link doesn't contain the text "youtube.com", this method will throw an exception
    - language: the language of the captions. If the caption doesn't exist

    Returns: a list of captions to display. 
    Unfortunately, this does not contain the speaker of the caption
    """
    files = [f for f in listdir(f"./{captions_folder}") if not isdir(f)]
    yt_video_id = getVideoId(yt_video)
    foundCaptions = False
    for file in files:
        if yt_video_id in file and language in file:
            foundCaptions = True
            captionsFile = file
            break

    if not foundCaptions:
        raise ValueError("We did not find the captions in the language you wanted. Try a different language or check if the captions were downloaded.")
    
    captions = []

    with open(f"{captions_folder}/{captionsFile}") as f:
        lines = f.readlines()
        # I noticed that the actual captions start on line 4 (0-indexed)
        i = 4
        while i < len(lines):
            nextline = lines[i]
            matches = re.findall('\d{2}:\d{2}:\d{2}.\d{3}', nextline)
            # print(matches)
            if matches:
                # this means this line is a timestamp
                nextcaption = {}
                nextcaption['start_time'] = convert_time_string(matches[0])
                nextcaption['end_time'] = convert_time_string(matches[1])
                nextcaption['caption'] = lines[i + 1][:-1]  # get rid of the newline
                captions.append(nextcaption)
                i += 2  # one line for times, one line for caption, newline gets split out
            i += 1
    
    return captions

##################################################
# Frame-related methods

def download_frames(yt_video, frame_rate=frame_rate_default):
    """
    Download frames from a youtube_video

    Parameters:
    - yt_video: link to the youtube_video
    - frame_rate (default value: 5): number of frames per second to capture

    REQUIRES: youtube-dl, ffmpeg
    """

    # first get video if its not already in the folder
    makeFolderIfNotExists(videos_folder)

    allVideos = [f for f in listdir(videos_folder) if not isdir(f)]
    videoId = getVideoId(yt_video)
    videoFound = False
    for video in allVideos:
        if videoId in video:
            filename = f"{videos_folder}/{video}"
            videoFound = True
            break
    
    if not videoFound:
        download_video_command = ["youtube-dl", "--output", f"{videos_folder}/%(title)s-%(id)s.%(ext)s", yt_video, "--print-json"]
        dl_json_bytes = subprocess.check_output(download_video_command)

        # get video's filename
        dl_json_string = dl_json_bytes.decode('utf-8')
        dl_json = json.loads(dl_json_string)
        filename = dl_json.get("_filename")[:-3] + "mkv"

    # then get frames using ffmpeg
    makeFolderIfNotExists(frames_folder)

    get_frames_command = ["ffmpeg", "-i", filename, "-vf", f"fps=fps={frame_rate}", f"{frames_folder}/frame%d_{frame_rate}fps_{videoId}.jpg"]
    get_frames_status = subprocess.check_call(get_frames_command)

    if get_frames_status:
        raise ModuleNotFoundError("Something went wrong with the ffmpeg command use here. \
            Maybe it was not installed? :/ Try reinstalling it on your machine!")

def get_frame(yt_video, time, frame_rate=frame_rate_default):
    """
    Gets the frame to display for a given time

    Parameters:
    - yt_video: the link to the youtube video to get captions for. 
    If this link doesn't contain the text "youtube.com", this method will throw an exception
    - time: The time since the start of the video, given in milliseconds
    - frame_rate (default = 5): the number of frames captured per second to parse through

    Returns: a 2-dimensional numpy array 
    containing the RGB values for each pixel of the image
    """
    
    framenumber = (int) (time / 1000 * frame_rate)
    frame_file_base = f"frame{framenumber}_{frame_rate}fps_{getVideoId(yt_video)}.jpg"
    frame_file = f"{frames_folder}/{frame_file_base}"

    frames = [f for f in listdir(frames_folder) if not isdir(f)]
    if frame_file_base not in frames:
        raise ValueError("The frame you wanted is not in the `frames` directory :( try someting else?")

    return imageio.imread(frame_file)

##################################################
# Utility Functions

def convert_time_string(time_string):
    """
    Converts a time string in the HH:MM:SS.MMM to a number of a seconds

    Parameters:
    - time_string: timestamp, given in the format HH:MM:SS.MMM, where
    H's are the digits for the number of hours, 
    M for minutes, S for seconds, and M for milliseconds

    Returns: integer containing the number of milliseconds
    """
    x = time.strptime(time_string.split('.')[0], '%H:%M:%S')
    return (int) (datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds() * 1000 + (int) (time_string.split('.')[1]))

def checkValidYoutubeVideo(yt_video):
    """
    Check if the given youtube video link is potentially a valid youtube link

    Parameters:
    - yt_video: the link of the youtube video

    Raises:
    - TypeError: if the video is not a string
    - ValueError: if the video is not a youtube link
    """
    if type(yt_video) != str:
        raise TypeError("The argument passed to this method is not a string. This is most likely a programmer error o.O")
    elif "youtube.com" not in yt_video:
        raise ValueError("The argument passed to this method does not contain 'youtube.com'. Maybe check if its a valid YouTube link?")

def getVideoId(yt_video):
    """
    Get a youtube video's ID

    Parameters:
    - yt_video: youtube link
     
    Returns: string containing the youtube video's ID
    """
    return yt_video.split("=")[1]

def makeFolderIfNotExists(folder_path):
    """
    Makes a folder at the specified path if the folder does not yet exist

    Parameters:
    - folder_path: directory to make a folder at
    """
    authorize_makefolder_command = ["sudo", "chmod", "+x", "./make_folder.sh"]
    subprocess.check_call(authorize_makefolder_command)
    
    check_folder_command = ["./make_folder.sh", folder_path]
    subprocess.check_call(check_folder_command)

##################################################
# Testing driver

if __name__ == "__main__":
    videolink = "https://www.youtube.com/watch?v=M7FIvfx5J10"
    download_captions(videolink)
    captionslist = get_caption_list(videolink)

    print(captionslist)

    frame_rate = 5
    download_frames(videolink, frame_rate)
    test_frame = get_frame(videolink, 5000, frame_rate)

    plt.imshow(test_frame)
    plt.show()