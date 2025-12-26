from pathlib import Path
import tempfile

import cv2
import numpy as np

from deepxromm.logging import logger


def validate_codec(codec: str, width: int = 640, height: int = 480) -> bool:
    """
    Validate if a video codec is available on the current system.

    Args:
        codec: FourCC codec code (e.g., "avc1", "DIVX", "XVID")
        width: Test video width (default 640)
        height: Test video height (default 480)

    Returns:
        True if codec is available and functional, False otherwise

    Note:
        Special cases "uncompressed" and 0 always return True as they
        use different encoding mechanisms.
    """
    # Special cases that don't use cv2.VideoWriter with fourcc
    if codec == "uncompressed" or codec == 0:
        return True

    # Test codec by creating a temporary VideoWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            test_path = Path(tmpdir) / "codec_test.avi"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(test_path),
                fourcc,
                1.0,  # Low FPS for test
                (width, height),
            )

            # Check if writer opened successfully
            is_valid = writer.isOpened()

            # Try writing a test frame to ensure codec actually works
            if is_valid:
                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(test_frame)

            writer.release()
            logger.debug(
                f"Codec validation for '{codec}': {'PASSED' if is_valid else 'FAILED'}"
            )
            return is_valid

        except Exception as e:
            logger.error(f"Codec validation for '{codec}' failed with exception: {e}")
            return False


def get_codec_error_message(failed_codec: str, operation: str) -> str:
    """
    Generate descriptive error message for codec validation failure.

    Args:
        failed_codec: The codec that failed validation
        operation: Operation type ("split" or "merge")

    Returns:
        Formatted error message with suggestions
    """
    return f"""
Video codec '{failed_codec}' is not available on this system for {operation}_rgb operation.

Common alternative codecs to try:
- "avc1"  : H.264 codec (best quality, not always available)
- "DIVX"  : DivX codec (widely available)
- "XVID"  : Xvid codec (widely available)
- "mp4v"  : MPEG-4 codec (generally available)
- "MJPG"  : Motion JPEG (always available, larger file sizes)
- "uncompressed" : Raw video via ffmpeg (largest files, highest quality)

To change the codec, update your project config file:
video_codec: "DIVX"  # Change this line to one of the alternatives above

Note: Codec availability depends on your OpenCV build and system codecs.
""".strip()
