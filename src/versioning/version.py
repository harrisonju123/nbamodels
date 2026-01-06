"""
Semantic versioning utilities for model versioning.

Follows semantic versioning spec (semver.org):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes (incompatible API/features)
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes (backwards compatible)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class BumpType(Enum):
    """Version bump type."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class Version:
    """Semantic version."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: 'Version') -> bool:
        """Compare versions."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __le__(self, other: 'Version') -> bool:
        return self < other or self == other

    def __gt__(self, other: 'Version') -> bool:
        return not (self <= other)

    def __ge__(self, other: 'Version') -> bool:
        return not (self < other)

    def __eq__(self, other: 'Version') -> bool:
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch)

    def bump(self, bump_type: BumpType) -> 'Version':
        """
        Create new version with bump applied.

        Args:
            bump_type: Type of version bump

        Returns:
            New version
        """
        if bump_type == BumpType.MAJOR:
            return Version(self.major + 1, 0, 0)
        elif bump_type == BumpType.MINOR:
            return Version(self.major, self.minor + 1, 0)
        elif bump_type == BumpType.PATCH:
            return Version(self.major, self.minor, self.patch + 1)
        else:
            raise ValueError(f"Unknown bump type: {bump_type}")


def parse_version(version_str: str) -> Version:
    """
    Parse version string to Version object.

    Args:
        version_str: Version string like "1.2.3" or "v1.2.3"

    Returns:
        Version object

    Raises:
        ValueError: If version string is invalid
    """
    # Remove leading 'v' if present
    version_str = version_str.lstrip('v')

    # Match semantic version pattern
    pattern = r'^(\d+)\.(\d+)\.(\d+)$'
    match = re.match(pattern, version_str)

    if not match:
        raise ValueError(f"Invalid version string: {version_str}")

    major, minor, patch = match.groups()
    return Version(int(major), int(minor), int(patch))


def infer_bump_type(description: str) -> BumpType:
    """
    Infer version bump type from description.

    Args:
        description: Description of changes

    Returns:
        Inferred bump type (defaults to PATCH)
    """
    description_lower = description.lower()

    # Check for major version indicators
    major_keywords = [
        'breaking', 'incompatible', 'major change',
        'rewrite', 'refactor all', 'architecture change'
    ]
    if any(kw in description_lower for kw in major_keywords):
        return BumpType.MAJOR

    # Check for minor version indicators
    minor_keywords = [
        'feature', 'add', 'new', 'enhancement',
        'improve', 'extended', 'support'
    ]
    if any(kw in description_lower for kw in minor_keywords):
        return BumpType.MINOR

    # Default to patch (bug fixes, small changes)
    return BumpType.PATCH


def get_next_version(
    current_version: Optional[str],
    bump_type: Optional[BumpType] = None,
    description: str = ""
) -> Version:
    """
    Get next version based on current version and bump type.

    Args:
        current_version: Current version string (None for first version)
        bump_type: Explicit bump type (None to infer from description)
        description: Description of changes (used to infer bump type)

    Returns:
        Next version
    """
    # Handle first version
    if current_version is None:
        return Version(1, 0, 0)

    # Parse current version
    current = parse_version(current_version)

    # Infer bump type if not provided
    if bump_type is None:
        bump_type = infer_bump_type(description)

    # Return bumped version
    return current.bump(bump_type)


def format_model_filename(model_name: str, version: Version) -> str:
    """
    Format model filename with version.

    Args:
        model_name: Base model name (e.g., 'spread_model')
        version: Version object

    Returns:
        Versioned filename (e.g., 'spread_model_v1.2.0.pkl')
    """
    return f"{model_name}_v{version}.pkl"


def parse_model_filename(filename: str) -> tuple[str, Version]:
    """
    Parse model filename to extract name and version.

    Args:
        filename: Filename like 'spread_model_v1.2.0.pkl'

    Returns:
        Tuple of (model_name, version)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Match pattern: {name}_v{version}.pkl
    pattern = r'^(.+)_v(\d+\.\d+\.\d+)\.pkl$'
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Invalid model filename: {filename}")

    model_name, version_str = match.groups()
    version = parse_version(version_str)

    return model_name, version
