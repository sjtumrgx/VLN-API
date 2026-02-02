"""Main application for embodied navigation system."""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
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

from .unified_analyzer import UnifiedAnalyzer
from .video_capture import VideoCapture
from .visualization import Visualizer
from .waypoint_generation import WaypointGenerator

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

        # Unified analyzer (combines scene analysis, task reasoning, waypoint generation)
        self.unified_analyzer = UnifiedAnalyzer(
            llm_client=self.llm_client,
            num_waypoints=self.config["waypoints"]["count"],
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
        )

    async def run(self):
        """Run the main navigation loop."""
        if self.offline_mode:
            await self._run_offline()
        else:
            await self._run_realtime()

    async def _run_realtime(self):
        """Run real-time processing mode (camera or live stream)."""
        logger.info("Starting real-time navigation system...")

        # Start video capture
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return

        self._running = True
        logger.info("Video capture started. Press 'q' to quit.")

        try:
            while self._running and not self._shutdown_event.is_set():
                await self._process_frame_realtime()
        except asyncio.CancelledError:
            logger.info("Navigation loop cancelled")
        finally:
            await self.shutdown()

    async def _run_offline(self):
        """Run offline video processing mode."""
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
        logger.info("Press 'q' to quit, any other key to continue to next frame.")

        self._running = True
        frame_idx = 0
        processed_count = 0

        try:
            while self._running and not self._shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video processing complete.")
                    break

                # Sample frames at interval
                if frame_idx % sample_interval == 0:
                    processed_count += 1
                    logger.info(f"Processing frame {frame_idx}/{total_frames} (#{processed_count})")

                    # Process and display frame
                    await self._process_frame_offline(frame, frame_idx, total_frames)

                frame_idx += 1

        except asyncio.CancelledError:
            logger.info("Video processing cancelled")
        finally:
            cap.release()
            await self.shutdown()

        logger.info(f"Processed {processed_count} frames from {total_frames} total frames")

    async def _process_frame_realtime(self):
        """Process a single frame in real-time mode."""
        # Get frame from capture
        frame = self.video_capture.get_frame(timeout=1.0)
        if frame is None:
            await asyncio.sleep(0.1)
            return

        # Letterbox frame to 640x640 for consistent processing
        frame, scale, (pad_w, pad_h) = letterbox(frame, target_size=(640, 640))

        try:
            # Run unified analysis (single API call)
            result = await self.unified_analyzer.analyze(
                frame=frame,
                task=self.goal,
                pad_h=pad_h,
            )

            logger.debug(f"Scene: {result.scene_analysis.summary}")
            logger.debug(f"Intent: {result.task_reasoning.intent}")

            # Log waypoints
            self.visualizer.log_waypoints(result.waypoints)

            # Render visualization
            vis_frame = self.visualizer.render(
                frame,
                scene_analysis=result.scene_analysis,
                task_reasoning=result.task_reasoning,
                waypoints=result.waypoints,
                waypoint_generator=self.waypoint_generator,
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            vis_frame = frame

        # Show frame
        key = self.visualizer.show(vis_frame)

        # Check for quit
        if key == ord('q') or key == ord('Q'):
            self._running = False

    async def _process_frame_offline(self, frame, frame_idx: int, total_frames: int):
        """Process a single frame in offline mode."""
        # Letterbox frame to 640x640 for consistent processing
        frame, scale, (pad_w, pad_h) = letterbox(frame, target_size=(640, 640))

        try:
            # Run unified analysis (single API call)
            result = await self.unified_analyzer.analyze(
                frame=frame,
                task=self.goal,
                pad_h=pad_h,
            )

            logger.info(f"Scene: {result.scene_analysis.summary}")
            logger.info(f"Intent: {result.task_reasoning.intent}")

            # Log waypoints
            self.visualizer.log_waypoints(result.waypoints)

            # Render visualization
            vis_frame = self.visualizer.render(
                frame,
                scene_analysis=result.scene_analysis,
                task_reasoning=result.task_reasoning,
                waypoints=result.waypoints,
                waypoint_generator=self.waypoint_generator,
            )

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            vis_frame = frame

        # Add frame info overlay
        info_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame and wait for key
        key = self.visualizer.show(vis_frame)

        # In offline mode, wait longer for user to see the result
        # Press 'q' to quit, any other key or timeout to continue
        if key == ord('q') or key == ord('Q'):
            self._running = False
        elif key == -1:
            # No key pressed within waitKey timeout, continue automatically
            await asyncio.sleep(0.5)  # Brief pause between frames

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
