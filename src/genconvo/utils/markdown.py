"""
Modified from
    https://github.com/HazyResearch/cartridges/blob/main/cartridges/contexts/finance/markdown.py
"""

from dataclasses import dataclass
from typing import List, Optional
import re
import os


@dataclass
class MarkdownSection:
    level: int
    title: str
    name: str
    path: str
    content: str
    desc: str = ""


def _to_camel_case(title: str) -> str:
    """Convert title to camel case for use as name."""
    name = title.replace("_", "__").replace(" ", "_").lower()
    return "".join(char for char in name if char.isalnum() or char == "_")


def _build_path(current_section: MarkdownSection, level: int, name: str) -> str:
    """Build path for new section based on current section and level."""
    if level > current_section.level:
        # Go deeper: add intermediate directories
        intermediate_dirs = ["_" for _ in range(level - current_section.level - 1)]
        return os.path.join(current_section.path, *intermediate_dirs, name)
    else:
        # Go shallower: truncate to appropriate level
        path_parts = current_section.path.split("/")[:level]
        return os.path.join(*path_parts, name)


def _is_header(line: str) -> tuple[bool, Optional[int], Optional[str]]:
    """Check if line is a header and return level and title if so."""
    stripped = line.lstrip()

    # Check for markdown headers (# Header)
    if header_match := re.match(r"^(#+)\s*(.*)", stripped):
        level = len(header_match.group(1))
        title = header_match.group(2).strip()
        return bool(title), level, title

    # Check for bold headers (**Header**)
    if (
        stripped.startswith("**")
        and stripped.endswith("**")
        and stripped.count("**") == 2
    ):
        title = stripped[2:-2].strip()
        return bool(title), None, title

    return False, None, None


def markdown_to_sections(text: str, root: str = "root") -> List[MarkdownSection]:
    """Parse markdown text into sections."""
    base_section = MarkdownSection(
        level=0, title=root, path=root, name=root, content=""
    )
    sections: List[MarkdownSection] = [base_section]
    active_sections: List[MarkdownSection] = [base_section]
    current_section = base_section

    # Determine bold header level based on max markdown header level
    lines = text.split("\n\n")
    max_level = 0
    for line in lines:
        if match := re.match(r"^(#+)\s", line.lstrip()):
            max_level = max(max_level, len(match.group(1)))
    bold_level = max_level + 1

    for line in lines:
        is_header, level, header_text = _is_header(line)

        if is_header and header_text:
            # Use bold_level for bold headers
            if level is None:
                level = bold_level

            name = _to_camel_case(header_text)
            path = _build_path(current_section, level, name)
            current_section = MarkdownSection(
                level=level, title=header_text, name=name, path=path, content=""
            )

            sections.append(current_section)
            # Keep only sections at deeper levels
            active_sections = [s for s in active_sections if s.level < level]
            active_sections.append(current_section)

        # Add line to all active sections
        for active_section in active_sections:
            active_section.content += line + "\n\n"

    # Filter out empty sections
    return [
        section for section in sections if section.content.replace("\n", "").strip()
    ]
