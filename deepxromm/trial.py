"""
Stores information about the Trial class, which is used for interacting with deepxromm trials
"""

from dataclasses import dataclass
from pathlib import Path

import blend_modes
import cv2
import numpy as np

from deepxromm.logging import logger


@dataclass
class Trial:
    """Interacts with deepxromm trials"""

    trial_path: Path

    # Read-only properties
    @property
    def trial_name(self):
        return self.trial_path.name

    # Public methods
    def find_cam_file(self, identifier: str, suffix: str | None = None) -> Path:
        """Find a video with identifier in its name in the current trial dir"""
        return self._find_file(".avi", identifier=identifier, suffix=suffix)

    def find_trial_csv(
        self, identifier: str | None = None, suffix: str | None = None
    ) -> Path:
        """
        Find the CSV pointsfile for a given trial or subpath with a certain identifier
        """
        return self._find_file(".csv", identifier=identifier, suffix=suffix)

    def make_rgb_video(self, codec: str, third_channel_mode: str = "difference"):
        """
        Takes the path to a trial subfolder and exports a single new video with
        cam1 video written to the red channel and cam2 video written to the
        green channel. The blue channel is, depending on the value of config
        "mode", either the difference blend between A and B, the multiply
        blend, or just a black frame.
        """
        logger.debug("Checking if RGB video already exists for {self.trial_path}...")
        rgb_video_path = self.trial_path / f"{self.trial_name}_rgb.avi"
        if rgb_video_path.exists():
            logger.warning("RGB video already created. Skipping.")
            return
        cam1_video_path = self.find_cam_file(identifier="cam1")
        cam1_video = cv2.VideoCapture(cam1_video_path)

        cam2_video_path = self.find_cam_file(identifier="cam2")
        cam2_video = cv2.VideoCapture(cam2_video_path)

        frame_width = int(cam1_video.get(3))
        frame_height = int(cam1_video.get(4))
        frame_rate = round(cam1_video.get(5), 2)

        # Note: "uncompressed" codec is not supported for merge_rgb
        # If needed in the future, implement ffmpeg pipeline like in split_rgb
        if codec == "uncompressed":
            raise RuntimeError(
                "The 'uncompressed' codec is not currently supported for merge_rgb operation. "
                "Please use a compressed codec like 'avc1', 'DIVX', 'XVID', 'mp4v', or 'MJPG'."
            )

        if codec == 0:
            fourcc = 0
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(rgb_video_path),
            fourcc,
            frame_rate,
            (frame_width, frame_height),
        )

        # Verify VideoWriter opened successfully
        if not out.isOpened():
            raise RuntimeError(
                f"Failed to create RGB video writer with codec '{codec}'"
            )

        i = 1
        while cam1_video.isOpened():
            if i == 1 or i % 50 == 0:
                logger.info(f"Current Frame: {i}")
            ret_cam1, frame_cam1 = cam1_video.read()
            _, frame_cam2 = cam2_video.read()
            if ret_cam1:
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2BGRA, 4).astype(
                    np.float32
                )
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2BGRA, 4).astype(
                    np.float32
                )
                frame_cam1 = cv2.normalize(
                    frame_cam1, None, 0, 255, norm_type=cv2.NORM_MINMAX
                )
                frame_cam2 = cv2.normalize(
                    frame_cam2, None, 0, 255, norm_type=cv2.NORM_MINMAX
                )
                if third_channel_mode == "difference":
                    extra_channel = blend_modes.difference(frame_cam1, frame_cam2, 1)
                elif third_channel_mode == "multiply":
                    extra_channel = blend_modes.multiply(frame_cam1, frame_cam2, 1)
                else:
                    extra_channel = np.zeros((frame_width, frame_height, 3), np.uint8)
                    extra_channel = cv2.cvtColor(
                        extra_channel, cv2.COLOR_BGR2BGRA, 4
                    ).astype(np.float32)
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2GRAY)
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2GRAY)
                extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGR2GRAY)
                merged = cv2.merge((extra_channel, frame_cam2, frame_cam1))
                out.write(merged)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

            i = i + 1
        cam1_video.release()
        cam2_video.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Merged RGB video created at {rgb_video_path}!")

    def _find_file(
        self,
        file_extension: str,
        identifier: str | None = None,
        suffix: str | None = None,
    ) -> Path:
        """
        Finds an arbitrary file within the given portion of a trial, given the file extension (including the '.') and any identifying characteristics
        """
        if suffix is None:
            path_to_search = self.trial_path
        else:
            path_to_search = self.trial_path / suffix

        all_files = list(path_to_search.glob("*"))
        logger.debug(all_files)
        if identifier is not None:
            files = list(path_to_search.glob(f"*{identifier}*{file_extension}"))
        else:
            files = list(path_to_search.glob(f"*{file_extension}"))

        logger.debug(files)
        if len(files) == 0:
            raise FileNotFoundError(
                f"No {file_extension} files containing '{identifier}' in {str(path_to_search)}"
            )
        if len(files) > 1:
            raise FileExistsError(
                f"Found more than 1 {file_extension} file containing '{identifier}' in {str(path_to_search)}"
            )

        return files[0]
