#!/bin/bash
if [ ! -d frames ]; then
    mkdir "frames"
fi
ffmpeg -i "$1" -vf fps=fps=5  "frames/frame%d.jpg"