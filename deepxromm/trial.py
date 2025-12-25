"""
Stores information about the Trial class, which is used for interacting with deepxromm trials
"""

from dataclasses import dataclass
from pathlib import Path


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

        if identifier is not None:
            files = list(path_to_search.glob(f"*{identifier}*{file_extension}"))
        else:
            files = list(path_to_search.glob(f"*.{file_extension}"))

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
