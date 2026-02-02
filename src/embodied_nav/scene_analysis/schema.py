"""JSON schema validation for scene analysis responses."""

SCENE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "bbox"],
            },
        },
        "obstacles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                },
                "required": ["label", "bbox"],
            },
        },
        "traversable_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                },
                "required": ["description", "bbox"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["objects", "obstacles", "traversable_regions", "summary"],
}


def validate_scene_analysis(data: dict) -> tuple[bool, list[str]]:
    """Validate scene analysis data against schema.

    Args:
        data: Parsed JSON data to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required top-level fields
    for field in ["objects", "obstacles", "traversable_regions", "summary"]:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate objects
    if "objects" in data:
        if not isinstance(data["objects"], list):
            errors.append("'objects' must be an array")
        else:
            for i, obj in enumerate(data["objects"]):
                obj_errors = _validate_object(obj, f"objects[{i}]")
                errors.extend(obj_errors)

    # Validate obstacles
    if "obstacles" in data:
        if not isinstance(data["obstacles"], list):
            errors.append("'obstacles' must be an array")
        else:
            for i, obs in enumerate(data["obstacles"]):
                obs_errors = _validate_obstacle(obs, f"obstacles[{i}]")
                errors.extend(obs_errors)

    # Validate traversable_regions
    if "traversable_regions" in data:
        if not isinstance(data["traversable_regions"], list):
            errors.append("'traversable_regions' must be an array")
        else:
            for i, region in enumerate(data["traversable_regions"]):
                region_errors = _validate_region(region, f"traversable_regions[{i}]")
                errors.extend(region_errors)

    # Validate summary
    if "summary" in data and not isinstance(data["summary"], str):
        errors.append("'summary' must be a string")

    return len(errors) == 0, errors


def _validate_bbox(bbox: dict, path: str) -> list[str]:
    """Validate bounding box."""
    errors = []
    if not isinstance(bbox, dict):
        return [f"{path}.bbox must be an object"]

    for field in ["x", "y", "width", "height"]:
        if field not in bbox:
            errors.append(f"{path}.bbox missing required field: {field}")
        elif not isinstance(bbox[field], (int, float)):
            errors.append(f"{path}.bbox.{field} must be a number")

    return errors


def _validate_object(obj: dict, path: str) -> list[str]:
    """Validate detected object."""
    errors = []
    if not isinstance(obj, dict):
        return [f"{path} must be an object"]

    if "label" not in obj:
        errors.append(f"{path} missing required field: label")
    elif not isinstance(obj["label"], str):
        errors.append(f"{path}.label must be a string")

    if "bbox" not in obj:
        errors.append(f"{path} missing required field: bbox")
    else:
        errors.extend(_validate_bbox(obj["bbox"], path))

    if "confidence" in obj:
        if not isinstance(obj["confidence"], (int, float)):
            errors.append(f"{path}.confidence must be a number")
        elif not 0 <= obj["confidence"] <= 1:
            errors.append(f"{path}.confidence must be between 0 and 1")

    return errors


def _validate_obstacle(obs: dict, path: str) -> list[str]:
    """Validate obstacle."""
    errors = []
    if not isinstance(obs, dict):
        return [f"{path} must be an object"]

    if "label" not in obs:
        errors.append(f"{path} missing required field: label")
    elif not isinstance(obs["label"], str):
        errors.append(f"{path}.label must be a string")

    if "bbox" not in obs:
        errors.append(f"{path} missing required field: bbox")
    else:
        errors.extend(_validate_bbox(obs["bbox"], path))

    return errors


def _validate_region(region: dict, path: str) -> list[str]:
    """Validate traversable region."""
    errors = []
    if not isinstance(region, dict):
        return [f"{path} must be an object"]

    if "description" not in region:
        errors.append(f"{path} missing required field: description")
    elif not isinstance(region["description"], str):
        errors.append(f"{path}.description must be a string")

    if "bbox" not in region:
        errors.append(f"{path} missing required field: bbox")
    else:
        errors.extend(_validate_bbox(region["bbox"], path))

    return errors
