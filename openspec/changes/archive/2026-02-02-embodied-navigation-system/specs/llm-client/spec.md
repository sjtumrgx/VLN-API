## ADDED Requirements

### Requirement: Support Gemini native API format
The system SHALL support calling Gemini API using the native v1beta format.

#### Scenario: Gemini native request
- **WHEN** user configures Gemini native format
- **THEN** system sends requests to `/v1beta/models/{model}:generateContent` with `contents` array

#### Scenario: Gemini native response parsing
- **WHEN** Gemini returns a response in native format
- **THEN** system extracts text from `candidates[0].content.parts[0].text`

### Requirement: Support OpenAI compatible API format
The system SHALL support calling LLM API using OpenAI compatible format.

#### Scenario: OpenAI compatible request
- **WHEN** user configures OpenAI compatible format
- **THEN** system sends requests to `/v1/chat/completions` with `messages` array

#### Scenario: OpenAI compatible response parsing
- **WHEN** API returns a response in OpenAI format
- **THEN** system extracts text from `choices[0].message.content`

### Requirement: Unified client interface
The system SHALL provide a unified interface regardless of the underlying API format.

#### Scenario: Format-agnostic usage
- **WHEN** application code calls the LLM client
- **THEN** the same method signature works for both Gemini and OpenAI formats

#### Scenario: Response normalization
- **WHEN** either API format returns a response
- **THEN** the client returns a normalized response object with consistent structure

### Requirement: Image input support
The system SHALL support sending images to the LLM for vision analysis.

#### Scenario: Send image with prompt
- **WHEN** user provides an image and a text prompt
- **THEN** system encodes the image as base64 and includes it in the request

#### Scenario: Image format handling
- **WHEN** image is provided in common formats (JPEG, PNG)
- **THEN** system correctly sets the MIME type in the request

### Requirement: Error handling and retry
The system SHALL handle API errors with configurable retry logic.

#### Scenario: Transient error retry
- **WHEN** API returns a 5xx error or timeout
- **THEN** system retries the request up to the configured maximum attempts

#### Scenario: Non-retryable error
- **WHEN** API returns a 4xx error (except 429)
- **THEN** system raises an exception without retry

#### Scenario: Rate limit handling
- **WHEN** API returns 429 (rate limited)
- **THEN** system waits for the specified backoff period before retry
