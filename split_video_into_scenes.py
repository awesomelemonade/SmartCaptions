#### FROM https://pyscenedetect.readthedocs.io/en/latest/examples/usage-python/
# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

import os

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)

# r"data\test\Sesame Street - Cookie Monster Sings C is for Cookie.mp4"
# scenes = find_scenes(os.path.join("data", "test", "battlefor.mp4"))
# print(scenes)
