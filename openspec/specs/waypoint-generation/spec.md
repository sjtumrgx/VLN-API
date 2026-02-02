## ADDED Requirements

### Requirement: Generate five navigation waypoints
The system SHALL generate exactly 5 waypoints for navigation path.

#### Scenario: Waypoint count
- **WHEN** waypoint generation completes
- **THEN** output contains exactly 5 waypoints

#### Scenario: Waypoint coordinates
- **WHEN** waypoints are generated
- **THEN** each waypoint includes `x` and `y` coordinates in pixel space

### Requirement: First waypoint at bottom center
The system SHALL place the first waypoint at the bottom center of the image.

#### Scenario: First waypoint position
- **WHEN** waypoints are generated
- **THEN** the first waypoint is at (image_width / 2, image_height)

### Requirement: Equal vertical spacing
The system SHALL distribute waypoints with equal vertical intervals.

#### Scenario: Vertical distribution
- **WHEN** waypoints are generated
- **THEN** the vertical distance between consecutive waypoints is equal

#### Scenario: Upward progression
- **WHEN** waypoints are ordered
- **THEN** waypoint 1 is at the bottom, waypoint 5 is highest (smallest y value)

### Requirement: Variable horizontal positioning
The system SHALL determine horizontal positions based on scene analysis and task reasoning.

#### Scenario: Obstacle avoidance
- **WHEN** an obstacle is detected in the direct path
- **THEN** waypoints curve horizontally to avoid the obstacle

#### Scenario: Clear path
- **WHEN** no obstacles are in the direct path
- **THEN** waypoints may remain near the center or follow the optimal route

### Requirement: Structured waypoint output
The system SHALL output waypoints in a structured JSON format.

#### Scenario: Waypoint output structure
- **WHEN** waypoint generation completes
- **THEN** output includes a `waypoints` array with 5 objects, each containing `x`, `y`, and `index` fields

#### Scenario: Waypoint ordering
- **WHEN** waypoints are output
- **THEN** waypoints are ordered by index from 1 (nearest/bottom) to 5 (farthest/top)
