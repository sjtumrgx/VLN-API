"""Unit tests for video capture module."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embodied_nav.video_capture.capture import VideoCapture


class TestVideoCapture:
    """Tests for VideoCapture class."""

    def test_init_with_camera_index(self):
        """Test initialization with camera index."""
        capture = VideoCapture(source=0)
        assert capture.is_camera is True
        assert capture.source == 0
        assert capture.queue_size == 1

    def test_init_with_file_path(self):
        """Test initialization with file path."""
        capture = VideoCapture(source="/path/to/video.mp4")
        assert capture.is_camera is False
        assert capture.source == "/path/to/video.mp4"

    def test_init_with_custom_queue_size(self):
        """Test initialization with custom queue size."""
        capture = VideoCapture(source=0, queue_size=3)
        assert capture.queue_size == 3

    @patch("cv2.VideoCapture")
    def test_start_opens_capture(self, mock_cv_capture):
        """Test that start() opens the video capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0)
        result = capture.start()

        assert result is True
        mock_cv_capture.assert_called_once_with(0)
        capture.stop()

    @patch("cv2.VideoCapture")
    def test_start_fails_if_not_opened(self, mock_cv_capture):
        """Test that start() returns False if capture fails to open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0)
        result = capture.start()

        assert result is False

    @patch("cv2.VideoCapture")
    def test_get_frame_returns_latest(self, mock_cv_capture):
        """Test that get_frame returns the latest frame."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, test_frame), (False, None)]
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0)
        capture.start()
        time.sleep(0.1)  # Allow capture thread to run

        frame = capture.get_frame(timeout=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        capture.stop()

    def test_get_frame_nowait_returns_none_when_empty(self):
        """Test that get_frame_nowait returns None when queue is empty."""
        capture = VideoCapture(source=0)
        frame = capture.get_frame_nowait()
        assert frame is None

    @patch("cv2.VideoCapture")
    def test_stop_releases_capture(self, mock_cv_capture):
        """Test that stop() releases the video capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0)
        capture.start()
        capture.stop()

        mock_cap.release.assert_called()
        assert capture.is_running is False

    @patch("cv2.VideoCapture")
    def test_disconnect_callback_called(self, mock_cv_capture):
        """Test that disconnect callback is called on camera disconnect."""
        mock_cap = MagicMock()
        # isOpened always returns True so we stay in the loop
        mock_cap.isOpened.return_value = True
        # First read succeeds, second fails (simulating disconnect)
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),  # This triggers disconnect
        ]
        mock_cv_capture.return_value = mock_cap

        callback = MagicMock()
        capture = VideoCapture(source=0, reconnect_attempts=1, reconnect_delay=0.01)
        capture.on_disconnect(callback)
        capture.start()
        time.sleep(0.3)

        callback.assert_called()
        capture.stop()

    @patch("cv2.VideoCapture")
    def test_end_callback_called_for_video_file(self, mock_cv_capture):
        """Test that end callback is called when video file ends."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv_capture.return_value = mock_cap

        callback = MagicMock()
        capture = VideoCapture(source="/path/to/video.mp4")
        capture.on_end(callback)

        with patch.object(Path, "exists", return_value=True):
            capture.start()
            time.sleep(0.1)

        callback.assert_called()
        capture.stop()

    @patch("cv2.VideoCapture")
    def test_get_frame_size(self, mock_cv_capture):
        """Test get_frame_size returns correct dimensions."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 640.0,  # CAP_PROP_FRAME_WIDTH
            4: 480.0,  # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0.0)
        mock_cap.read.return_value = (False, None)
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0)
        capture.start()

        size = capture.get_frame_size()
        assert size == (640, 480)
        capture.stop()

    @patch("cv2.VideoCapture")
    def test_queue_discards_old_frames(self, mock_cv_capture):
        """Test that queue discards old frames when full."""
        frames = [np.full((480, 640, 3), i, dtype=np.uint8) for i in range(5)]
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        mock_cv_capture.return_value = mock_cap

        capture = VideoCapture(source=0, queue_size=1)
        capture.start()
        time.sleep(0.2)  # Allow all frames to be processed

        # Should get the latest frame
        frame = capture.get_frame_nowait()
        # Queue should have at most 1 frame
        assert capture._frame_queue.qsize() <= 1
        capture.stop()
