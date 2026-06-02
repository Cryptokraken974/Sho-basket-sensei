"use strict";

// Standard half-/full-court reference coordinates (feet) for a 94x50 court.
// Used to pre-fill court coordinates when a coach places a landmark; values
// remain editable per calibration.
const DEFAULT_COURT = {
  baseline_left_sideline: [0, 0],
  baseline_right_sideline: [0, 50],
  opposite_baseline_left_sideline: [94, 0],
  opposite_baseline_right_sideline: [94, 50],
  free_throw_left_lane_intersection: [19, 17],
  free_throw_right_lane_intersection: [19, 33],
  opposite_free_throw_left_lane_intersection: [75, 17],
  opposite_free_throw_right_lane_intersection: [75, 33],
  center_circle_center: [47, 25],
  center_circle_top: [47, 31],
  center_circle_bottom: [47, 19],
  three_point_arc_left_wing: [14, 3],
  three_point_arc_right_wing: [14, 47],
  three_point_arc_top: [29, 25],
  opposite_three_point_arc_left_wing: [80, 3],
  opposite_three_point_arc_right_wing: [80, 47],
  opposite_three_point_arc_top: [65, 25],
};

function fmtNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "?";
  return Math.round(Number(value) * 100) / 100;
}

window.bvGamePage = function (config) {
  return {
    gameId: config.gameId,
    videoId: config.videoId,
    video: null,
    uploading: false,
    error: "",
    pollTimer: null,
    overlays: { detection_boxes: true, track_trails: true, ball_path: true },
    activeRunId: null,
    runIdInput: "",
    progress: null,
    detections: [],
    tracks: [],
    rafId: null,
    sourceW: 0,
    sourceH: 0,
    overlayFps: 30,
    demoRunning: false,
    analyzing: false,
    engineNote: "YOLO+ByteTrack detects players & ball (needs the CV runtime). Demo uses synthetic data.",
    clickMode: false,
    clickLabel: "Player 1",
    clickPrompts: [],
    clickTracking: false,

    fmt: fmtNumber,

    // Prefer the transcoded 720p proxy; fall back to the uploaded original so
    // playback (and the demo overlay) work even without ffmpeg.
    get playSrc() {
      if (!this.video) return "";
      if (this.video.state === "ready" && this.video.proxy_url) return this.video.proxy_url;
      return this.video.original_url || "";
    },
    get usingOriginal() {
      return !!this.video && this.video.state !== "ready" && !!this.video.original_url;
    },
    get canPlay() {
      return !!this.playSrc;
    },
    get hasOverlayData() {
      return this.detections.length > 0 || this.tracks.length > 0;
    },

    async init() {
      if (this.videoId) await this.refresh();
    },

    async refresh() {
      try {
        const res = await fetch(`/api/videos/${this.videoId}`);
        if (!res.ok) return;
        this.video = await res.json();
        if (this.video.state === "uploaded" || this.video.state === "processing") {
          this.schedulePoll();
        } else if (this.pollTimer) {
          clearTimeout(this.pollTimer);
          this.pollTimer = null;
        }
      } catch (e) {
        this.error = String(e);
      }
    },

    schedulePoll() {
      if (this.pollTimer) clearTimeout(this.pollTimer);
      this.pollTimer = setTimeout(() => this.refresh(), 2000);
    },

    async upload(event) {
      const input = event.target.querySelector('input[type="file"]');
      if (!input || !input.files.length) return;
      this.uploading = true;
      this.error = "";
      const body = new FormData();
      body.append("file", input.files[0]);
      try {
        const res = await fetch(`/api/games/${this.gameId}/video?process_inline=false`, {
          method: "POST",
          body,
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Upload failed (${res.status})`);
        }
        this.video = await res.json();
        this.videoId = this.video.id;
        this.schedulePoll();
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.uploading = false;
      }
    },

    onVideoReady() {
      const video = this.$refs.video;
      const canvas = this.$refs.overlay;
      if (!video || !canvas) return;
      const resize = () => {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      };
      resize();
      window.addEventListener("resize", resize);
      if (this.rafId) cancelAnimationFrame(this.rafId);
      const loop = () => {
        this.drawOverlay();
        this.rafId = requestAnimationFrame(loop);
      };
      loop();
    },

    currentFrame() {
      const video = this.$refs.video;
      return video ? Math.round(video.currentTime * this.overlayFps) : 0;
    },

    // Pixel space the loaded detections live in (demo: virtual 1000-space;
    // real runs: the played video's native resolution).
    sourceWidth() {
      return this.sourceW || (this.$refs.video ? this.$refs.video.videoWidth : 0);
    },
    sourceHeight() {
      return this.sourceH || (this.$refs.video ? this.$refs.video.videoHeight : 0);
    },

    drawOverlay() {
      const video = this.$refs.video;
      const canvas = this.$refs.overlay;
      if (!video || !canvas || !video.videoWidth) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const srcW = this.sourceWidth() || video.videoWidth;
      const srcH = this.sourceHeight() || video.videoHeight;
      const sx = canvas.width / srcW;
      const sy = canvas.height / srcH;
      const frame = this.currentFrame();

      if (this.overlays.detection_boxes) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#3ccf91";
        ctx.font = "12px system-ui";
        ctx.fillStyle = "#3ccf91";
        for (const d of this.detections) {
          if (d.frame_idx !== frame) continue;
          const b = d.bbox;
          ctx.strokeRect(b.x * sx, b.y * sy, b.width * sx, b.height * sy);
          ctx.fillText(`${d.class_name} ${fmtNumber(d.confidence)}`, b.x * sx, b.y * sy - 4);
        }
      }

      if (this.overlays.track_trails) {
        ctx.fillStyle = "#ff7a3c";
        for (const t of this.tracks) {
          if (t.frame_idx > frame) continue;
          const b = t.bbox;
          ctx.globalAlpha = t.frame_idx === frame ? 0.9 : 0.25;
          const cx = (b.x + b.width / 2) * sx;
          const cy = (b.y + b.height / 2) * sy;
          ctx.beginPath();
          ctx.arc(cx, cy, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      if (this.overlays.ball_path) {
        const balls = this.detections.filter((d) => d.class_name === "ball" && d.frame_idx <= frame);
        ctx.strokeStyle = "#f5b042";
        ctx.lineWidth = 2;
        ctx.beginPath();
        balls.forEach((d, i) => {
          const cx = (d.bbox.x + d.bbox.width / 2) * sx;
          const cy = (d.bbox.y + d.bbox.height / 2) * sy;
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.stroke();
      }
    },

    async loadRunPair(detectionRunId, trackingRunId) {
      const [pRes, dRes, tRes] = await Promise.all([
        fetch(`/api/runs/${detectionRunId}/progress`),
        fetch(`/api/runs/${detectionRunId}/detections`),
        fetch(`/api/runs/${trackingRunId}/tracks`),
      ]);
      this.activeRunId = detectionRunId;
      this.progress = pRes.ok ? await pRes.json() : null;
      this.detections = dRes.ok ? await dRes.json() : [];
      this.tracks = tRes.ok ? await tRes.json() : [];
    },

    async runAnalysis() {
      this.analyzing = true;
      this.error = "";
      try {
        const res = await fetch(`/api/videos/${this.videoId}/analysis`, { method: "POST" });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Analysis failed (${res.status})`);
        }
        const body = await res.json();
        this.sourceW = body.source_width;
        this.sourceH = body.source_height;
        this.overlayFps = body.fps;
        this.engineNote = `${body.engine}: ${body.object_count} detections across ${body.frame_count} frames`;
        await this.loadRunPair(body.detection_run_id, body.tracking_run_id);
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.analyzing = false;
      }
    },

    onCanvasClick(event) {
      if (!this.clickMode) return;
      const video = this.$refs.video;
      const canvas = this.$refs.overlay;
      if (!video || !canvas || !video.videoWidth) return;
      const rect = canvas.getBoundingClientRect();
      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      // Clicks are stored in the played file's native pixel space, which is the
      // same space SAM 2 will track in (it runs on that file).
      this.clickPrompts.push({
        object_label: this.clickLabel || `object ${this.clickPrompts.length + 1}`,
        frame_idx: this.currentFrame(),
        x: nx * video.videoWidth,
        y: ny * video.videoHeight,
      });
    },

    async runClickTrack() {
      if (this.clickPrompts.length === 0) return;
      this.clickTracking = true;
      this.error = "";
      try {
        const res = await fetch(`/api/videos/${this.videoId}/click-track`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompts: this.clickPrompts }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Click-track failed (${res.status})`);
        }
        const body = await res.json();
        this.sourceW = body.source_width;
        this.sourceH = body.source_height;
        this.overlayFps = body.fps;
        this.engineNote = `${body.engine}: tracked ${body.object_count} boxes across ${body.frame_count} frames`;
        await this.loadRunPair(body.detection_run_id, body.tracking_run_id);
        this.clickMode = false;
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.clickTracking = false;
      }
    },

    async runDemo() {
      this.demoRunning = true;
      this.error = "";
      try {
        const video = this.$refs.video;
        const params = new URLSearchParams();
        if (video && Number.isFinite(video.duration) && video.duration > 0) {
          params.set("duration_s", String(video.duration));
        }
        const res = await fetch(
          `/api/videos/${this.videoId}/demo-analysis?${params.toString()}`,
          { method: "POST" }
        );
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Demo analysis failed (${res.status})`);
        }
        const body = await res.json();
        this.sourceW = body.source_width;
        this.sourceH = body.source_height;
        this.overlayFps = body.fps;
        await this.loadRunPair(body.detection_run_id, body.tracking_run_id);
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.demoRunning = false;
      }
    },

    async loadRun(event) {
      event.preventDefault();
      const runId = this.runIdInput.trim();
      if (!runId) return;
      this.error = "";
      try {
        const [pRes, dRes, tRes] = await Promise.all([
          fetch(`/api/runs/${runId}/progress`),
          fetch(`/api/runs/${runId}/detections`),
          fetch(`/api/runs/${runId}/tracks`),
        ]);
        if (!pRes.ok) throw new Error(`Run ${runId} not found`);
        this.activeRunId = runId;
        // Real runs are in the played video's native pixel space.
        this.sourceW = 0;
        this.sourceH = 0;
        const video = this.$refs.video;
        this.overlayFps = this.video && this.video.fps ? this.video.fps : 30;
        if (video && video.videoWidth) {
          this.sourceW = video.videoWidth;
          this.sourceH = video.videoHeight;
        }
        this.progress = await pRes.json();
        this.detections = dRes.ok ? await dRes.json() : [];
        this.tracks = tRes.ok ? await tRes.json() : [];
        if (this.progress.status === "running" || this.progress.status === "queued") {
          setTimeout(() => this.loadRunProgress(runId), 2000);
        }
      } catch (e) {
        this.error = String(e.message || e);
      }
    },

    async loadRunProgress(runId) {
      const res = await fetch(`/api/runs/${runId}/progress`);
      if (res.ok) {
        this.progress = await res.json();
        if (this.progress.status === "running" || this.progress.status === "queued") {
          setTimeout(() => this.loadRunProgress(runId), 2000);
        }
      }
    },
  };
};

window.bvCalibration = function (config) {
  return {
    videoId: config.videoId,
    proxyUrl: `/api/videos/${config.videoId}/proxy_720p.mp4`,
    landmarks: [],
    selectedLandmark: "",
    captured: false,
    capturedTs: 0,
    points: [],
    saving: false,
    error: "",
    savedId: "",

    fmt: fmtNumber,

    async init() {
      try {
        const res = await fetch("/api/landmarks");
        if (res.ok) {
          this.landmarks = await res.json();
          this.selectedLandmark = this.landmarks[0] || "";
        }
      } catch (e) {
        this.error = String(e);
      }
    },

    capture() {
      const video = this.$refs.video;
      const canvas = this.$refs.frame;
      if (!video || !video.videoWidth) {
        this.error = "Let the video load before capturing.";
        return;
      }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      this.capturedTs = video.currentTime;
      this.captured = true;
      this.error = "";
    },

    placePoint(event) {
      const canvas = this.$refs.frame;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const imageX = (event.clientX - rect.left) * scaleX;
      const imageY = (event.clientY - rect.top) * scaleY;
      const court = DEFAULT_COURT[this.selectedLandmark] || [0, 0];
      this.points = this.points.filter((p) => p.landmark !== this.selectedLandmark);
      this.points.push({
        landmark: this.selectedLandmark,
        image_x: imageX,
        image_y: imageY,
        court_x: court[0],
        court_y: court[1],
      });
    },

    async save() {
      if (this.points.length < 4) return;
      this.saving = true;
      this.error = "";
      this.savedId = "";
      try {
        const res = await fetch(`/api/videos/${this.videoId}/calibrations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            point_pairs: this.points,
            valid_from_s: 0.0,
            valid_to_s: null,
            created_by: "coach",
          }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Save failed (${res.status})`);
        }
        const body = await res.json();
        this.savedId = body.calibration_id;
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.saving = false;
      }
    },
  };
};
