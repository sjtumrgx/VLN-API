"""Main application for embodied navigation system."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml

from .llm_client import GeminiNativeClient, OpenAICompatibleClient
from .scene_analysis import SceneAnalyzer
from .task_reasoning import TaskReasoner
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
        api_format: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        goal: str = "Navigate forward safely",
    ):
        """Initialize the navigation system.

        Args:
            config_path: Path to config file
            source: Video source (camera index or file path)
            api_format: API format ("gemini" or "openai")
            api_key: API key (overrides config)
            model: Model name (overrides config)
            goal: Navigation goal
        """
        self.goal = goal
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Load config
        self.config = self._load_config(config_path)

        # Override config with arguments
        if source is not None:
            self.config["video"]["source"] = source
        if api_format:
            self.config["api"]["format"] = api_format
        if api_key:
            self.config["api"][api_format]["api_key"] = api_key
        if model:
            self.config["api"][api_format]["model"] = model

        # Initialize components
        self._init_components()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        default_config = {
            "api": {
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

        # LLM client
        api_format = self.config["api"]["format"]
        api_config = self.config["api"][api_format]

        if api_format == "gemini":
            self.llm_client = GeminiNativeClient(
                base_url=api_config["base_url"],
                api_key=api_config["api_key"],
                model=api_config["model"],
            )
        else:
            self.llm_client = OpenAICompatibleClient(
                base_url=api_config["base_url"],
                api_key=api_config["api_key"],
                model=api_config["model"],
            )

        # Scene analyzer
        self.scene_analyzer = SceneAnalyzer(self.llm_client)

        # Task reasoner
        self.task_reasoner = TaskReasoner(self.llm_client)

        # Waypoint generator
        self.waypoint_generator = WaypointGenerator(
            llm_client=self.llm_client,
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
        logger.info("Starting embodied navigation system...")

        # Start video capture
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return

        self._running = True
        logger.info("Video capture started. Press 'q' to quit.")

        try:
            while self._running and not self._shutdown_event.is_set():
                await self._process_frame()
        except asyncio.CancelledError:
            logger.info("Navigation loop cancelled")
        finally:
            await self.shutdown()

    async def _process_frame(self):
        """Process a single frame."""
        # Get frame from capture
        frame = self.video_capture.get_frame(timeout=1.0)
        if frame is None:
            await asyncio.sleep(0.1)
            return

        image_size = (frame.shape[1], frame.shape[0])

        try:
            # Run scene analysis
            scene_analysis = await self.scene_analyzer.analyze(frame)
            logger.debug(f"Scene: {scene_analysis.summary}")

            # Run task reasoning
            task_reasoning = await self.task_reasoner.reason(
                scene_analysis,
                goal=self.goal,
            )
            logger.debug(f"Intent: {task_reasoning.intent}")

            # Generate waypoints
            waypoint_result = await self.waypoint_generator.generate(
                image_size=image_size,
                scene_analysis=scene_analysis,
                task_reasoning=task_reasoning,
            )

            # Log waypoints
            self.visualizer.log_waypoints(waypoint_result.waypoints)

            # Render visualization
            vis_frame = self.visualizer.render(
                frame,
                scene_analysis=scene_analysis,
                task_reasoning=task_reasoning,
                waypoints=waypoint_result.waypoints,
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
        "--api-format", "-f",
        type=str,
        choices=["gemini", "openai"],
        default="gemini",
        help="API format to use",
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
        help="Navigation goal",
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
        api_format=args.api_format,
        api_key=args.api_key,
        model=args.model,
        goal=args.goal,
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
