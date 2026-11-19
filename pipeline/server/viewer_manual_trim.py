"""Install the visual manual episode-trim panel into the pinned Viewer.

The Viewer is patched at startup from a pinned upstream checkout. The manual
trim panel belongs to the default Episode tab, directly below the video so
operators can watch and trim at the same time. This patch is idempotent across
restarts and migrates the panel from its earlier playback-bar position.
"""

from pathlib import Path

MARKER = "// bodypipeline: manual episode trim v2"
LEGACY_MARKER_V1 = "// bodypipeline: manual episode trim v1"
COMPONENT = "src/components/manual-trim.tsx"
VIEWER = "src/app/[org]/[dataset]/[episode]/episode-viewer.tsx"
IMPORT_ANCHOR = (
    'import { TaskAnnotationStudio } from "@/components/task-annotation-studio";'
)
IMPORT = 'import { ManualTrimPanel } from "@/components/manual-trim";'
PANEL = "              <ManualTrimPanel />\n"
VIDEO_ANCHOR = "              )}\n\n              {/* Language Instruction */}"
VIDEO_REPLACEMENT = (
    "              )}\n"
    "\n"
    "              <ManualTrimPanel />\n"
    "\n"
    "              {/* Language Instruction */}"
)


def patch_manual_trim(cache: Path):
    source = cache / "source"
    page = source / VIEWER
    text = page.read_text()

    text = text.replace(LEGACY_MARKER_V1, MARKER)
    if IMPORT not in text:
        if text.count(IMPORT_ANCHOR) != 1:
            raise RuntimeError(
                "Unexpected Visualizer source; cannot install manual episode trim"
            )
        text = text.replace(
            IMPORT_ANCHOR,
            IMPORT_ANCHOR + "\n" + ("" if MARKER in text else MARKER + "\n") + IMPORT,
        )
    elif MARKER not in text:
        text = text.replace(IMPORT, MARKER + "\n" + IMPORT)

    # Remove the panel wherever an earlier patch placed it, then install one
    # copy below the video. This also migrates persistent Viewer source caches.
    text = text.replace(PANEL, "")
    text = text.replace(
        "              )}\n\n\n              {/* Language Instruction */}",
        VIDEO_ANCHOR,
    )
    if VIDEO_REPLACEMENT not in text:
        if text.count(VIDEO_ANCHOR) != 1:
            raise RuntimeError(
                "Unexpected Visualizer source; cannot place manual episode trim"
            )
        text = text.replace(VIDEO_ANCHOR, VIDEO_REPLACEMENT)
    page.write_text(text)
