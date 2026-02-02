## 1. Project Setup

- [x] 1.1 Create project directory structure (src/, tests/, config/)
- [x] 1.2 Initialize Python project with pyproject.toml or requirements.txt
- [x] 1.3 Add dependencies: opencv-python, httpx, matplotlib, numpy, Pillow
- [x] 1.4 Create configuration file for API endpoints and credentials

## 2. Video Capture Module

- [x] 2.1 Implement VideoCapture class with camera and file source support
- [x] 2.2 Implement single-frame queue with thread-safe access
- [x] 2.3 Add frame extraction method for on-demand retrieval
- [x] 2.4 Implement disconnection detection and reconnection logic
- [x] 2.5 Add unit tests for video capture module

## 3. LLM Client Module

- [x] 3.1 Implement base LLMClient interface with unified method signatures
- [x] 3.2 Implement GeminiNativeClient for v1beta format
- [x] 3.3 Implement OpenAICompatibleClient for /v1/chat/completions format
- [x] 3.4 Add image encoding (base64) and MIME type handling
- [x] 3.5 Implement retry logic with exponential backoff
- [x] 3.6 Add response normalization to unified format
- [x] 3.7 Add unit tests for LLM client module

## 4. Scene Analysis Module

- [x] 4.1 Design prompt template for scene analysis
- [x] 4.2 Implement SceneAnalyzer class that calls LLM with image
- [x] 4.3 Parse LLM response into structured format (objects, obstacles, traversable_regions)
- [x] 4.4 Add JSON schema validation for response
- [x] 4.5 Add unit tests for scene analysis module

## 5. Task Reasoning Module

- [x] 5.1 Design prompt template for task reasoning
- [x] 5.2 Implement TaskReasoner class that takes scene analysis and goal
- [x] 5.3 Parse LLM response into structured format (task_understanding, intent, reasoning)
- [x] 5.4 Add unit tests for task reasoning module

## 6. Waypoint Generation Module

- [x] 6.1 Implement WaypointGenerator class
- [x] 6.2 Calculate first waypoint at bottom center of image
- [x] 6.3 Calculate equal vertical spacing for 5 waypoints
- [x] 6.4 Integrate LLM output for horizontal positioning
- [x] 6.5 Output structured waypoint array with x, y, index
- [x] 6.6 Add unit tests for waypoint generation module

## 7. Visualization Module

- [x] 7.1 Implement Visualizer class with OpenCV window
- [x] 7.2 Add bounding box drawing for objects and obstacles
- [x] 7.3 Add text overlay for task reasoning display
- [x] 7.4 Implement waypoint markers with green-to-red color gradient
- [x] 7.5 Implement smooth curve drawing (spline interpolation) connecting waypoints
- [x] 7.6 Add coordinate display/logging for waypoints
- [x] 7.7 Add unit tests for visualization module

## 8. Main Application

- [x] 8.1 Implement main loop with asyncio + thread pool architecture
- [x] 8.2 Wire up all modules: capture → LLM → analysis → reasoning → waypoints → visualization
- [x] 8.3 Add command-line argument parsing (source, API format, model, etc.)
- [x] 8.4 Add graceful shutdown handling (Ctrl+C)
- [x] 8.5 Add logging configuration

## 9. Integration Testing

- [x] 9.1 Test end-to-end flow with sample video file
- [x] 9.2 Test with live camera input
- [x] 9.3 Test both Gemini native and OpenAI compatible API formats
- [x] 9.4 Verify visualization output matches specifications
