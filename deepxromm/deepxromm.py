"""
Defines deepxromm's public API
"""

from pathlib import Path


from deepxromm.analyzer import Analyzer
from deepxromm.autocorrector import Autocorrector
from deepxromm.dlc_config import DlcConfig
from deepxromm.network import Network
from deepxromm.project import Project, ProjectFactory
from deepxromm.xma_data_processor import XMADataProcessor
from deepxromm.augmenter import Augmenter


class DeepXROMM:
    """A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking"""

    project: Project
    dlc_config: DlcConfig
    _analyzer: Analyzer
    _autocorrector: Autocorrector
    _network: Network
    _data_processor: XMADataProcessor
    _augmenter: Augmenter

    def __init__(self):
        # Prevent direct instantiation because we need to do different
        # initialization work in create_ vs load_ methods.
        raise NotImplementedError("Use create_new_project or load_project instead.")

    @classmethod
    def create_new_project(
        cls,
        working_dir: str | Path = Path.cwd(),
        experimenter="NA",
        mode="2D",
        codec="avc1",
    ):
        """Create a new deepxromm project"""
        deepxromm = DeepXROMM.__new__(DeepXROMM)
        if isinstance(working_dir, str):
            working_dir = Path(working_dir)
        deepxromm.project = ProjectFactory.create_new_config(
            working_dir, experimenter, mode, codec
        )
        deepxromm._analyzer = Analyzer(deepxromm.project)
        deepxromm._autocorrector = Autocorrector(deepxromm.project)
        deepxromm._network = Network(deepxromm.project)
        deepxromm._data_processor = XMADataProcessor(
            deepxromm.project, deepxromm.project.dlc_config
        )
        deepxromm._augmenter = Augmenter(deepxromm.project)
        return deepxromm

    @classmethod
    def load_project(cls, working_dir: str | Path):
        """Create a new xrommtools project"""
        deepxromm = DeepXROMM.__new__(DeepXROMM)
        deepxromm.project = ProjectFactory.load_config(working_dir)
        deepxromm._analyzer = Analyzer(deepxromm.project)
        deepxromm._autocorrector = Autocorrector(deepxromm.project)
        deepxromm._network = Network(deepxromm.project)
        deepxromm._data_processor = XMADataProcessor(
            deepxromm.project, deepxromm.project.dlc_config
        )
        deepxromm._augmenter = Augmenter(deepxromm.project)
        return deepxromm

    def xma_to_dlc(self):
        """Converts XMAlab-compatible data to DLC format"""
        self._network.xma_to_dlc()

    def create_training_dataset(self):
        """Creates a training dataset based on current project data"""
        self._network.create_training_dataset()

    def train_network(self, **kwargs):
        """Starts training the network using data in the working directory."""
        self._network.train(**kwargs)

    def analyze_videos(self):
        """Analyze videos with a pre-existing network"""
        self._analyzer.analyze_videos()

    def dlc_to_xma(self):
        """Convert DLC output from training to XMA format"""
        self._data_processor.dlc_to_xma()

    def extract_outlier_frames(self, **kwargs):
        """Extract outlier frames for re-analysis from DLC output"""
        self._augmenter.extract_outlier_frames(**kwargs)

    def merge_datasets(self, **kwargs):
        """Create a refined dataset that includes the data collected from the outliers in outliers.yaml for each trial"""
        self._augmenter.merge_datasets(**kwargs)

    def autocorrect_trials(self):
        """Do XMAlab-style autocorrect on the tracked beads for all trials"""
        self._autocorrector.autocorrect_trials()

    def get_dlc_bodyparts(self):
        """Pull the names of the XMAlab markers from the 2Dpoints file"""
        return self.dlc_config.get_dlc_bodyparts()

    def split_rgb(self, trial_path, codec=None):
        """Takes a RGB video with different grayscale data written to the R, G,
        and B channels and splits it back into its component source videos.

        Args:
            trial_path: Path to trial directory containing RGB video
            codec: Video codec to use. If None, uses video_codec from config.
                   Common options: "avc1", "DIVX", "XVID", "mp4v", "MJPG", "uncompressed"

        Raises:
            RuntimeError: If the specified codec is not available on this system
        """
        return self._data_processor.split_rgb(trial_path, codec)

    def analyze_video_similarity_project(self):
        """Analyze all videos in a project and take their average similar. This
        is dangerous, as it will assume that all cam1/cam2 pairs match or don't match!
        """
        return self._analyzer.analyze_video_similarity_project()

    def analyze_video_similarity_trial(self, **kwargs):
        """Analyze the average similarity between trials using image hashing"""
        return self._analyzer.analyze_video_similarity_trial(**kwargs)

    def get_max_dissimilarity_for_trial(self, trial_path, window):
        """Calculate the dissimilarity within the trial given the frame sliding window."""
        return self._analyzer.get_max_dissimilarity_for_trial(trial_path, window)

    def analyze_marker_similarity_project(self):
        """Analyze all videos in a project and get their average rhythmicity. This assumes
        that all cam1/2 pairs are either the same or different!"""
        return self._analyzer.analyze_marker_similarity_project()

    def analyze_marker_similarity_trial(self):
        """Analyze marker similarity for a pair of trials. Returns the mean difference for
        paired marker positions (X - X, Y - Y for each marker)"""
        return self._analyzer.analyze_marker_similarity_trial()

    @staticmethod
    def train_many_projects(parent_dir):
        """Train and analyze multiple deepxromm projects given a parent folder"""
        parent_path = Path(parent_dir)
        for folder in parent_path.iterdir():
            if not folder.is_dir():
                continue
            project_path = folder
            deepxromm = DeepXROMM.load_project(str(project_path))
            deepxromm.xma_to_dlc()
            deepxromm.create_training_dataset()
            deepxromm.train_network()
            deepxromm.analyze_videos()
            deepxromm.dlc_to_xma()
