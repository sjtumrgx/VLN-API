## ADDED Requirements

### Requirement: Support multiple video sources
The system SHALL support both camera devices and video files as input sources.

#### Scenario: Camera input
- **WHEN** user specifies a camera device index (e.g., 0)
- **THEN** system opens the camera and starts capturing frames

#### Scenario: Video file input
- **WHEN** user specifies a video file path
- **THEN** system opens the video file and starts reading frames

### Requirement: Single-frame queue buffering
The system SHALL maintain a queue with maximum length of 1 for frame buffering.

#### Scenario: New frame arrives
- **WHEN** a new frame is captured and the queue already contains a frame
- **THEN** the old frame is discarded and replaced with the new frame

#### Scenario: Frame retrieval
- **WHEN** a consumer requests a frame from the queue
- **THEN** the most recent frame is returned without blocking capture

### Requirement: Frame extraction on demand
The system SHALL provide frames to consumers on demand without blocking the capture thread.

#### Scenario: Consumer requests frame
- **WHEN** the LLM client requests a frame for analysis
- **THEN** the system returns the latest frame from the queue immediately

#### Scenario: No frame available
- **WHEN** a frame is requested but the queue is empty
- **THEN** the system waits until a frame becomes available or timeout occurs

### Requirement: Graceful source disconnection handling
The system SHALL detect and handle video source disconnection gracefully.

#### Scenario: Camera disconnected
- **WHEN** the camera device becomes unavailable during capture
- **THEN** the system emits a disconnection event and attempts reconnection

#### Scenario: Video file ends
- **WHEN** the video file reaches the end
- **THEN** the system emits an end-of-stream event
