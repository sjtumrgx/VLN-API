"""Main application for embodied navigation system."""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from .llm_client import GeminiNativeClient, OpenAICompatibleClient, letterbox


@dataclass
class Profile:
    """API configuration profile."""

    name: str
    base_url: str
    api_key: str
    model: str
    format: str  # "gemini" or "openai"

from .unified_analyzer import UnifiedAnalyzer, UnifiedAnalysisResult
from .video_capture import VideoCapture
from .visualization import Visualizer
from .waypoint_generation import Waypoint, WaypointGenerator

logger = logging.getLogger(__name__)


class EmbodiedNavigationSystem:
    """Main application class for embodied navigation."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        source: Optional[str] = None,
        profile: Optional[str] = None,
        api_format: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        goal: str = "Navigate forward safely",
        task: Optional[str] = None,
        offline: bool = False,
        sample_interval: Optional[int] = None,
        export: Optional[bool] = None,
        export_path: Optional[str] = None,
        export_speed: Optional[float] = None,
    ):
        """Initialize the navigation system.

        Args:
            config_path: Path to config file
            source: Video source (camera index or file path)
            profile: API profile name to use (from config file)
            api_format: API format ("gemini" or "openai"), overrides profile
            api_key: API key (overrides profile)
            model: Model name (overrides profile)
            goal: Navigation goal
            task: Custom task instruction (overrides goal)
            offline: Enable offline video processing mode
            sample_interval: Frame sampling interval (default: auto-detect from FPS)
            export: Enable video export (overrides config)
            export_path: Output video path (overrides auto-generated name)
            export_speed: Playback speed multiplier (overrides config)
        """
        self.goal = task if task else goal
        self.offline_mode = offline
        self.sample_interval = sample_interval
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Load config
        self.config = self._load_config(config_path)

        # Load and select profile
        profiles = self._load_profiles()
        selected_profile = self._get_profile(profile, profiles)
        logger.info(f"Using API profile: {selected_profile.name}")

        # Apply CLI overrides to profile
        if api_format:
            selected_profile.format = api_format
        if api_key:
            selected_profile.api_key = api_key
        if model:
            selected_profile.model = model

        # Store resolved profile for component initialization
        self._active_profile = selected_profile

        # Override video source if provided
        if source is not None:
            self.config["video"]["source"] = source

        # Auto-detect offline mode for video files
        source_val = self.config["video"]["source"]
        if isinstance(source_val, str) and not source_val.isdigit():
            if Path(source_val).exists() and Path(source_val).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                self.offline_mode = True
                logger.info(f"Auto-detected video file, enabling offline mode")

        # Video export settings
        export_config = self.config.get("export", {})
        self.export_enabled = export if export is not None else export_config.get("enabled", False)
        self.export_path = export_path
        self.export_speed = export_speed if export_speed is not None else export_config.get("speed", 1.0)
        self.export_output_dir = export_config.get("output_dir", "output")
        self.export_codec = export_config.get("codec", "mp4v")
        self.export_fps = export_config.get("fps", None)
        self._video_writer = None

        # Initialize components
        self._init_components()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        default_config = {
            "api": {
                "default_profile": "default",
                "profiles": [
                    {
                        "name": "default",
                        "base_url": "http://38.207.171.242:8317",
                        "api_key": "",
                        "model": "gemini-3-flash-preview",
                        "format": "gemini",
                    },
                ],
                # Legacy format support (for backward compatibility)
                "format": "gemini",
                "gemini": {
                    "base_url": "http://38.207.171.242:8317",
                    "model": "gemini-3-flash-preview",
                    "api_key": "",
                },
                "openai": {
                    "base_url": "http://38.207.171.242:8317",
                    "model": "gemini-2.5-pro",
                    "api_key": "",
                },
            },
            "video": {
                "source": 0,
                "queue_size": 1,
                "reconnect_attempts": 3,
                "reconnect_delay": 1.0,
            },
            "image": {
                "target_size": [640, 640],
            },
            "export": {
                "enabled": False,
                "output_dir": "output",
                "speed": 1.0,
                "codec": "mp4v",
                "fps": None,
            },
            "waypoints": {
                "count": 5,
            },
            "visualization": {
                "window_name": "Embodied Navigation",
                "show_coordinates": True,
                "near_color": [0, 255, 0],
                "far_color": [0, 0, 255],
            },
            "logging": {
                "level": "INFO",
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._deep_merge(default_config, loaded)

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_profiles(self) -> List[Profile]:
        """Parse profiles from config, supporting both new and legacy formats."""
        api_config = self.config["api"]

        # Check if new profiles format exists
        if "profiles" in api_config and api_config["profiles"]:
            profiles = []
            for i, p in enumerate(api_config["profiles"]):
                # Validate required fields
                required_fields = ["name", "base_url", "api_key", "model", "format"]
                missing = [f for f in required_fields if f not in p or p[f] is None]
                if missing:
                    raise ValueError(
                        f"Profile at index {i} is missing required fields: {', '.join(missing)}"
                    )
                if p["format"] not in ("gemini", "openai"):
                    raise ValueError(
                        f"Profile '{p['name']}' has invalid format '{p['format']}'. Must be 'gemini' or 'openai'."
                    )
                profiles.append(Profile(
                    name=p["name"],
                    base_url=p["base_url"],
                    api_key=p["api_key"],
                    model=p["model"],
                    format=p["format"],
                ))
            return profiles

        # Legacy format: convert gemini/openai sections to implicit profiles
        profiles = []
        if "gemini" in api_config:
            g = api_config["gemini"]
            profiles.append(Profile(
                name="gemini",
                base_url=g.get("base_url", ""),
                api_key=g.get("api_key", ""),
                model=g.get("model", ""),
                format="gemini",
            ))
        if "openai" in api_config:
            o = api_config["openai"]
            profiles.append(Profile(
                name="openai",
                base_url=o.get("base_url", ""),
                api_key=o.get("api_key", ""),
                model=o.get("model", ""),
                format="openai",
            ))
        return profiles

    def _get_profile(self, profile_name: Optional[str], profiles: List[Profile]) -> Profile:
        """Select profile by name with error handling.

        Args:
            profile_name: Name of profile to select, or None for default
            profiles: List of available profiles

        Returns:
            Selected Profile

        Raises:
            ValueError: If profile not found
        """
        if not profiles:
            raise ValueError("No profiles configured")

        # Use default profile if none specified
        if profile_name is None:
            profile_name = self.config["api"].get("default_profile")
            # For legacy config, use the format as profile name
            if profile_name is None:
                profile_name = self.config["api"].get("format", "gemini")

        # Find profile by name
        for profile in profiles:
            if profile.name == profile_name:
                return profile

        # Profile not found - raise error with available names
        available = [p.name for p in profiles]
        raise ValueError(
            f"Profile '{profile_name}' not found. Available profiles: {', '.join(available)}"
        )

    def _init_components(self):
        """Initialize all components."""
        # Video capture
        video_config = self.config["video"]
        source = video_config["source"]
        # Convert string number to int for camera
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.video_capture = VideoCapture(
            source=source,
            queue_size=video_config["queue_size"],
            reconnect_attempts=video_config["reconnect_attempts"],
            reconnect_delay=video_config["reconnect_delay"],
        )

        # LLM client - use resolved profile
        profile = self._active_profile

        if profile.format == "gemini":
            self.llm_client = GeminiNativeClient(
                base_url=profile.base_url,
                api_key=profile.api_key,
                model=profile.model,
            )
        else:
            self.llm_client = OpenAICompatibleClient(
                base_url=profile.base_url,
                api_key=profile.api_key,
                model=profile.model,
            )

        # Image processing settings
        image_config = self.config.get("image", {})
        target_size = image_config.get("target_size", [640, 640])
        self.target_size = tuple(target_size)

        # Unified analyzer (combines scene analysis, task reasoning, waypoint generation)
        jpeg_quality = image_config.get("jpeg_quality", 95)
        self.unified_analyzer = UnifiedAnalyzer(
            llm_client=self.llm_client,
            num_waypoints=self.config["waypoints"]["count"],
            jpeg_quality=jpeg_quality,
        )

        # Waypoint generator (for smooth curve rendering only)
        self.waypoint_generator = WaypointGenerator(
            num_waypoints=self.config["waypoints"]["count"],
        )

        # Visualizer
        vis_config = self.config["visualization"]
        self.visualizer = Visualizer(
            window_name=vis_config["window_name"],
            show_coordinates=vis_config["show_coordinates"],
            near_color=tuple(vis_config["near_color"]),
            far_color=tuple(vis_config["far_color"]),
            left_box_width=vis_config.get("left_box_width", 0.22),
            right_box_width=vis_config.get("right_box_width", 0.22),
        )

        # Translated user task (will be set after first API call)
        self.user_task_english = None

        # Shared state for decoupled VLM/display loops (P0)
        self._latest_result: Optional[UnifiedAnalysisResult] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pad_h: int = 0
        self._vlm_running: bool = False

        # Interpolation state (P2)
        self._prev_result: Optional[UnifiedAnalysisResult] = None
        self._result_timestamp: float = 0.0
        self._blend_duration: float = 0.5  # seconds to blend between results

        # Frame change detection state (P3)
        image_config = self.config.get("image", {})
        self._change_threshold: float = image_config.get("change_threshold", 15.0)
        self._max_staleness: float = image_config.get("max_staleness", 5.0)
        self._last_vlm_frame: Optional[np.ndarray] = None
        self._last_vlm_time: float = 0.0

    async def run(self):
        """Run the main navigation loop."""
        if self.offline_mode:
            await self._run_offline()
        else:
            await self._run_realtime()

    async def _run_realtime(self):
        """Run real-time processing mode with decoupled VLM and display loops."""
        logger.info("Starting real-time navigation system...")

        # Start video capture
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return

        self._running = True
        logger.info("Video capture started. Press 'q' to quit.")

        try:
            # Run VLM and display loops concurrently
            await asyncio.gather(
                self._vlm_loop_realtime(),
                self._display_loop_realtime(),
            )
        except asyncio.CancelledError:
            logger.info("Navigation loop cancelled")
        finally:
            await self.shutdown()

    async def _vlm_loop_realtime(self):
        """VLM inference loop: capture frame -> call VLM -> update shared state."""
        loop = asyncio.get_event_loop()
        while self._running and not self._shutdown_event.is_set():
            # Get frame from capture (non-blocking via executor)
            frame = await loop.run_in_executor(
                None, self.video_capture.get_frame, 1.0
            )
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            # Letterbox frame for consistent processing
            frame, scale, (pad_w, pad_h) = letterbox(frame, target_size=self.target_size)

            # Update latest frame for display loop
            self._latest_frame = frame
            self._latest_pad_h = pad_h

            # Skip VLM if scene hasn't changed significantly (P3)
            if not self._should_call_vlm(frame):
                continue

            try:
                self._vlm_running = True
                result = await self.unified_analyzer.analyze(
                    frame=frame,
                    task=self.goal,
                    pad_h=pad_h,
                )
                self._vlm_running = False
                self._last_vlm_frame = frame.copy()
                self._last_vlm_time = time.monotonic()

                logger.debug(f"Scene: {result.scene_summary}")
                logger.debug(f"Intent: {result.intent}")
                self.visualizer.log_waypoints(result.waypoints)

                # Update translated task if available
                if result.task_english and not self.user_task_english:
                    self.user_task_english = result.task_english

                # Save previous result for interpolation (P2)
                self._prev_result = self._latest_result
                self._result_timestamp = time.monotonic()
                self._latest_result = result

            except Exception as e:
                self._vlm_running = False
                logger.error(f"Error in VLM loop: {e}")

    async def _display_loop_realtime(self):
        """Display loop: render latest frame with latest VLM result at ~30fps."""
        while self._running and not self._shutdown_event.is_set():
            frame = self._latest_frame
            result = self._get_display_result()
            pad_h = self._latest_pad_h

            if frame is None:
                await asyncio.sleep(1 / 30)
                continue

            # Render visualization with latest VLM result overlay
            if result is not None:
                vis_frame = self.visualizer.render(
                    frame.copy(),
                    scene_summary=result.scene_summary,
                    task_understanding=result.task_understanding,
                    intent=result.intent,
                    waypoints=result.waypoints,
                    waypoint_generator=self.waypoint_generator,
                    linear_velocity=result.linear_velocity,
                    angular_velocity=result.angular_velocity,
                    pad_h=pad_h,
                    user_task=self.user_task_english or result.task_english,
                )
            else:
                vis_frame = frame.copy()

            # Show frame
            key = self.visualizer.show(vis_frame)
            if key == ord('q') or key == ord('Q'):
                self._running = False

            await asyncio.sleep(1 / 30)

    async def _run_offline(self):
        """Run offline video processing with decoupled VLM and display loops."""
        source = self.config["video"]["source"]
        logger.info(f"Starting offline video processing: {source}")

        # Open video file directly with OpenCV
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {source}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine sample interval (default: sample 1 frame per second)
        if self.sample_interval is not None:
            sample_interval = self.sample_interval
        else:
            sample_interval = max(1, int(fps))  # 1 frame per second

        logger.info(f"Video FPS: {fps}, Total frames: {total_frames}, Sample interval: {sample_interval}")
        logger.info(f"Task: {self.goal}")
        logger.info("Press 'q' to quit.")

        # Initialize video writer if export is enabled
        if self.export_enabled:
            self._init_video_writer(source, fps, sample_interval)

        self._running = True

        # Shared state for offline frame reading
        self._offline_cap = cap
        self._offline_fps = fps
        self._offline_total_frames = total_frames
        self._offline_sample_interval = sample_interval
        self._offline_frame_idx = 0
        self._offline_done = False

        try:
            await asyncio.gather(
                self._vlm_loop_offline(),
                self._display_loop_offline(),
            )
        except asyncio.CancelledError:
            logger.info("Video processing cancelled")
        finally:
            cap.release()
            self._close_video_writer()
            await self.shutdown()

    async def _vlm_loop_offline(self):
        """VLM loop for offline mode: read frames and run VLM asynchronously."""
        cap = self._offline_cap
        sample_interval = self._offline_sample_interval
        total_frames = self._offline_total_frames
        processed_count = 0

        while self._running and not self._shutdown_event.is_set():
            if self._offline_done:
                break

            # Read next frame (blocking I/O via executor)
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.info("Video processing complete.")
                self._offline_done = True
                break

            frame_idx = self._offline_frame_idx
            self._offline_frame_idx += 1

            # Letterbox every frame for display
            lb_frame, scale, (pad_w, pad_h) = letterbox(frame, target_size=self.target_size)
            self._latest_frame = lb_frame
            self._latest_pad_h = pad_h

            # Only run VLM on sampled frames
            if frame_idx % sample_interval != 0:
                continue

            # Skip VLM if scene hasn't changed significantly (P3)
            if not self._should_call_vlm(lb_frame):
                processed_count += 1
                logger.info(f"Skipped frame {frame_idx}/{total_frames} (#{processed_count}, scene unchanged)")
                continue

            processed_count += 1
            logger.info(f"Processing frame {frame_idx}/{total_frames} (#{processed_count})")

            try:
                self._vlm_running = True
                result = await self.unified_analyzer.analyze(
                    frame=lb_frame,
                    task=self.goal,
                    pad_h=pad_h,
                )
                self._vlm_running = False
                self._last_vlm_frame = lb_frame.copy()
                self._last_vlm_time = time.monotonic()

                logger.info(f"Scene: {result.scene_summary}")
                logger.info(f"Intent: {result.intent}")
                self.visualizer.log_waypoints(result.waypoints)

                if result.task_english and not self.user_task_english:
                    self.user_task_english = result.task_english

                # Save previous result for interpolation (P2)
                self._prev_result = self._latest_result
                self._result_timestamp = time.monotonic()
                self._latest_result = result

            except Exception as e:
                self._vlm_running = False
                logger.error(f"Error processing frame {frame_idx}: {e}")

        logger.info(f"VLM processed {processed_count} frames from {total_frames} total")

    async def _display_loop_offline(self):
        """Display loop for offline mode: render every frame with latest VLM overlay."""
        frame_delay = 1.0 / max(1.0, self._offline_fps)

        while self._running and not self._shutdown_event.is_set():
            frame = self._latest_frame
            result = self._get_display_result()
            pad_h = self._latest_pad_h

            if frame is None:
                if self._offline_done:
                    break
                await asyncio.sleep(frame_delay)
                continue

            # Render visualization with latest VLM result overlay
            if result is not None:
                vis_frame = self.visualizer.render(
                    frame.copy(),
                    scene_summary=result.scene_summary,
                    task_understanding=result.task_understanding,
                    intent=result.intent,
                    waypoints=result.waypoints,
                    waypoint_generator=self.waypoint_generator,
                    linear_velocity=result.linear_velocity,
                    angular_velocity=result.angular_velocity,
                    pad_h=pad_h,
                    user_task=self.user_task_english or result.task_english,
                )
            else:
                vis_frame = frame.copy()

            # Add frame info overlay
            info_text = f"Frame: {self._offline_frame_idx}/{self._offline_total_frames}"
            cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Write frame to video if export is enabled
            self._write_frame(vis_frame)

            # Show frame
            key = self.visualizer.show(vis_frame)
            if key == ord('q') or key == ord('Q'):
                self._running = False

            if self._offline_done and self._latest_frame is not None:
                # Video ended, stop display loop
                break

            await asyncio.sleep(frame_delay)

    def _init_video_writer(self, source: str, source_fps: float, sample_interval: int):
        """Initialize video writer for export.

        In decoupled mode, every displayed frame is written, so output FPS
        matches the source FPS (adjusted by speed multiplier).
        """
        # Determine output path
        if self.export_path:
            output_path = self.export_path
        else:
            # Auto-generate name: video_name + model + speed
            source_path = Path(source)
            video_name = source_path.stem
            model_name = self._active_profile.model.replace("/", "-").replace(":", "-")
            speed_str = f"{self.export_speed}x" if self.export_speed != 1.0 else "1x"
            output_name = f"{video_name}_{model_name}_{speed_str}.mp4"

            # Ensure output directory exists
            output_dir = Path(self.export_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / output_name)

        # Calculate output FPS
        if self.export_fps:
            output_fps = self.export_fps
        else:
            # Base FPS is source_fps / sample_interval (frames we actually process)
            # Then multiply by speed
            base_fps = source_fps / sample_interval
            output_fps = base_fps * self.export_speed

        # Get frame size from target_size
        width, height = self.target_size

        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*self.export_codec)
        self._video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if self._video_writer.isOpened():
            logger.info(f"Video export enabled: {output_path} (FPS: {output_fps:.1f}, Speed: {self.export_speed}x)")
        else:
            logger.error(f"Failed to create video writer: {output_path}")
            self._video_writer = None

    def _close_video_writer(self):
        """Close video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            logger.info("Video export completed")
            self._video_writer = None

    def _write_frame(self, frame: np.ndarray):
        """Write frame to video file."""
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.write(frame)

    def _should_call_vlm(self, frame: np.ndarray) -> bool:
        """Check if VLM should be called based on frame change detection.

        Args:
            frame: Current letterboxed frame

        Returns:
            True if VLM should be called, False to skip
        """
        # Always call for the first frame
        if self._last_vlm_frame is None:
            return True

        # Disabled if threshold is 0
        if self._change_threshold <= 0:
            return True

        # Force call if results are stale
        if self._max_staleness > 0:
            elapsed = time.monotonic() - self._last_vlm_time
            if elapsed >= self._max_staleness:
                logger.debug("VLM call forced due to staleness timer")
                return True

        # Compute mean absolute difference between frames
        diff = cv2.absdiff(frame, self._last_vlm_frame)
        mean_diff = diff.mean()

        if mean_diff < self._change_threshold:
            logger.debug(f"Frame change {mean_diff:.1f} < threshold {self._change_threshold}, skipping VLM")
            return False

        return True

    def _get_display_result(self) -> Optional[UnifiedAnalysisResult]:
        """Get the interpolated result for display, blending between prev and current."""
        current = self._latest_result
        prev = self._prev_result
        if current is None:
            return None
        if prev is None:
            return current

        # Calculate blend factor based on elapsed time since result update
        elapsed = time.monotonic() - self._result_timestamp
        blend = min(1.0, elapsed / self._blend_duration)

        if blend >= 1.0:
            return current

        return self._interpolate_results(prev, current, blend)

    def _interpolate_results(
        self,
        prev: UnifiedAnalysisResult,
        current: UnifiedAnalysisResult,
        t: float,
    ) -> UnifiedAnalysisResult:
        """Interpolate between two VLM results.

        Args:
            prev: Previous VLM result
            current: Current VLM result
            t: Blend factor 0.0 (all prev) to 1.0 (all current)

        Returns:
            Interpolated result
        """
        # Lerp waypoint x-coordinates (y is fixed by vertical position)
        interp_waypoints = []
        num_wp = min(len(prev.waypoints), len(current.waypoints))
        for i in range(num_wp):
            wp_prev = prev.waypoints[i]
            wp_cur = current.waypoints[i]
            x = int(wp_prev.x + t * (wp_cur.x - wp_prev.x))
            y = int(wp_prev.y + t * (wp_cur.y - wp_prev.y))
            interp_waypoints.append(Waypoint(index=wp_cur.index, x=x, y=y))

        # If current has more waypoints, append them directly
        for i in range(num_wp, len(current.waypoints)):
            interp_waypoints.append(current.waypoints[i])

        # Lerp velocities
        linear_v = prev.linear_velocity + t * (current.linear_velocity - prev.linear_velocity)
        angular_v = prev.angular_velocity + t * (current.angular_velocity - prev.angular_velocity)

        # Strings snap to current immediately
        return UnifiedAnalysisResult(
            scene_summary=current.scene_summary,
            task_understanding=current.task_understanding,
            intent=current.intent,
            reasoning=current.reasoning,
            waypoints=interp_waypoints,
            linear_velocity=linear_v,
            angular_velocity=angular_v,
            task_english=current.task_english,
            raw_response=current.raw_response,
        )

    async def shutdown(self):
        """Shutdown the system gracefully."""
        logger.info("Shutting down...")
        self._running = False

        # Stop video capture
        self.video_capture.stop()

        # Close LLM client
        await self.llm_client.close()

        # Close visualizer
        self.visualizer.close()

        logger.info("Shutdown complete")

    def request_shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_event.set()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Embodied Navigation System")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Video source (camera index or file path)",
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        help="API profile name to use (from config file)",
    )
    parser.add_argument(
        "--api-format", "-f",
        type=str,
        choices=["gemini", "openai"],
        default=None,
        help="API format to use (overrides profile)",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name",
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        default="Navigate forward safely",
        help="Navigation goal (default mode)",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Custom task instruction (overrides --goal)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable offline video processing mode (auto-detected for video files)",
    )
    parser.add_argument(
        "--sample-interval", "-i",
        type=int,
        help="Frame sampling interval for offline mode (default: 1 frame per second)",
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Enable video export (saves processed video to file)",
    )
    parser.add_argument(
        "--export-path", "-o",
        type=str,
        help="Output video path (default: auto-generated from source name + model + speed)",
    )
    parser.add_argument(
        "--export-speed",
        type=float,
        help="Playback speed multiplier for exported video (default: from config)",
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create system
    system = EmbodiedNavigationSystem(
        config_path=args.config,
        source=args.source,
        profile=args.profile,
        api_format=args.api_format,
        api_key=args.api_key,
        model=args.model,
        goal=args.goal,
        task=args.task,
        offline=args.offline,
        sample_interval=args.sample_interval,
        export=args.export if args.export else None,
        export_path=args.export_path,
        export_speed=args.export_speed,
    )

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        system.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    asyncio.run(system.run())


if __name__ == "__main__":
    main()
