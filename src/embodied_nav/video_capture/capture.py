"""Video capture with camera and file source support."""

import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional, Union

import cv2
import numpy as np


class VideoCapture:
    """Video capture class supporting camera devices and video files."""

    def __init__(
        self,
        source: Union[int, str, Path],
        queue_size: int = 1,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        """Initialize video capture.

        Args:
            source: Camera index (int) or video file path (str/Path)
            queue_size: Maximum frames to buffer (default 1)
            reconnect_attempts: Number of reconnection attempts on disconnect
            reconnect_delay: Delay between reconnection attempts in seconds
        """
        self.source = source
        self.queue_size = queue_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._frame_queue: Queue = Queue(maxsize=queue_size)
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._is_camera = isinstance(source, int)
        self._disconnected = threading.Event()
        self._end_of_stream = threading.Event()

        self._on_disconnect_callback = None
        self._on_end_callback = None

    @property
    def is_camera(self) -> bool:
        """Return True if source is a camera device."""
        return self._is_camera

    @property
    def is_running(self) -> bool:
        """Return True if capture is running."""
        return self._running

    def start(self) -> bool:
        """Start video capture.

        Returns:
            True if capture started successfully, False otherwise
        """
        if self._running:
            return True

        if not self._open_capture():
            return False

        self._running = True
        self._disconnected.clear()
        self._end_of_stream.clear()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        return True

    def stop(self):
        """Stop video capture."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        with self._lock:
            if self._capture:
                self._capture.release()
                self._capture = None

        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except:
                break

    def get_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get the latest frame from the queue.

        Args:
            timeout: Maximum time to wait for a frame (None = block forever)

        Returns:
            Frame as numpy array, or None if timeout or stopped
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except:
            return None

    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """Get the latest frame without blocking.

        Returns:
            Frame as numpy array, or None if no frame available
        """
        try:
            return self._frame_queue.get_nowait()
        except:
            return None

    def on_disconnect(self, callback):
        """Set callback for disconnect events."""
        self._on_disconnect_callback = callback

    def on_end(self, callback):
        """Set callback for end-of-stream events."""
        self._on_end_callback = callback

    def _open_capture(self) -> bool:
        """Open the video capture device/file."""
        with self._lock:
            if self._capture:
                self._capture.release()

            if self._is_camera:
                self._capture = cv2.VideoCapture(self.source)
            else:
                path = str(self.source)
                if not Path(path).exists():
                    return False
                self._capture = cv2.VideoCapture(path)

            return self._capture.isOpened()

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self._running:
            with self._lock:
                if not self._capture or not self._capture.isOpened():
                    break
                ret, frame = self._capture.read()

            if not ret:
                if self._is_camera:
                    self._handle_disconnect()
                else:
                    self._handle_end_of_stream()
                break

            # Update queue with latest frame (discard old if full)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except:
                    pass
            self._frame_queue.put(frame)

    def _handle_disconnect(self):
        """Handle camera disconnection with reconnection attempts."""
        self._disconnected.set()
        if self._on_disconnect_callback:
            self._on_disconnect_callback()

        for attempt in range(self.reconnect_attempts):
            if not self._running:
                break
            time.sleep(self.reconnect_delay)
            if self._open_capture():
                self._disconnected.clear()
                self._capture_loop()  # Resume capture
                return

    def _handle_end_of_stream(self):
        """Handle end of video file."""
        self._end_of_stream.set()
        self._running = False
        if self._on_end_callback:
            self._on_end_callback()

    def get_frame_size(self) -> Optional[tuple]:
        """Get frame dimensions (width, height)."""
        with self._lock:
            if self._capture:
                width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (width, height)
        return None
