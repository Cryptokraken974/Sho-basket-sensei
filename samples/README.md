# Sample videos

Short `.mp4` clips for trying the app locally live here.

## Trying a sample

1. Start the app:

   ```bash
   uv run uvicorn basketvision_coach.api:app --reload
   ```

2. Open <http://localhost:8000/>, create a game, and upload your clip from this
   folder with the **Upload video** button.

3. On the game page, click **▶ Run demo analysis**. This fabricates synthetic
   detections and runs them through the real detection + tracking pipeline, so
   you can see player boxes, track trails, and the ball path drawn over your
   video and toggle each overlay layer.

> The demo uses synthetic detections because the project does not yet ship a
> trained object detector. Wiring a real detector (e.g. YOLO) is a follow-up;
> the tracking, calibration projection, and overlay rendering it would feed are
> already in place.

If `ffmpeg`/`ffprobe` are installed the app transcodes a 720p proxy for
playback; otherwise it falls back to playing your uploaded original directly.
