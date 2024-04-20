"""A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking"""
from pathlib import Path
from ruamel.yaml.comments import CommentedMap

from .analyzer import Analyzer
from .autocorrector import Autocorrector
from .network import Network
from .project import Project
from .xma_data_processor import XMADataProcessor


class DeepXROMM:
    """A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking"""

    config: CommentedMap
    _analyzer: Analyzer
    _autocorrector: Autocorrector
    _network: Network
    _data_processor: XMADataProcessor

    def __init__(self):
        # Prevent direct instantiation because we need to do different
        # initialization work in create_ vs load_ methods.
        raise NotImplementedError("Use create_new_project or load_project instead.")

    @classmethod
    def create_new_project(cls, working_dir=None, experimenter="NA", mode="2D"):
        '''Create a new xrommtools project'''
        deepxromm = DeepXROMM.__new__(DeepXROMM)
        deepxromm.config = Project.create_new_config(working_dir, experimenter, mode)
        deepxromm._analyzer = Analyzer(deepxromm.config)
        deepxromm._autocorrector = Autocorrector(deepxromm.config)
        deepxromm._network = Network(deepxromm.config)
        deepxromm._data_processor = XMADataProcessor(deepxromm.config)
        return deepxromm

    @classmethod
    def load_project(cls, working_dir=None):
        '''Create a new xrommtools project'''
        deepxromm = DeepXROMM.__new__(DeepXROMM)
        deepxromm.config = Project.load_config(working_dir)
        deepxromm._analyzer = Analyzer(deepxromm.config)
        deepxromm._autocorrector = Autocorrector(deepxromm.config)
        deepxromm._network = Network(deepxromm.config)
        deepxromm._data_processor = XMADataProcessor(deepxromm.config)
        return deepxromm

    def train_network(self):
        '''Starts training the network using xrommtools-compatible data in the working directory.'''
        self._network.train()

    def analyze_videos(self):
        '''Analyze videos with a pre-existing network'''
        self._analyzer.analyze_videos()

    def autocorrect_trial(self):
        '''Do XMAlab-style autocorrect on the tracked beads'''
        self._autocorrector.autocorrect_trial()

    def get_bodyparts_from_xma(self, path_to_trial, mode):
        '''Pull the names of the XMAlab markers from the 2Dpoints file'''
        return self._data_processor.get_bodyparts_from_xma(path_to_trial, mode)

    def split_rgb(self, trial_path, codec='avc1'):
        '''Takes a RGB video with different grayscale data written to the R, G,
        and B channels and splits it back into its component source videos.'''
        return self._data_processor.split_rgb(trial_path, codec)

    def analyze_video_similarity_project(self):
        '''Analyze all videos in a project and take their average similar. This
        is dangerous, as it will assume that all cam1/cam2 pairs match or don't match!'''
        return self._analyzer.analyze_video_similarity_project()

    def analyze_video_similarity_trial(self):
        '''Analyze the average similarity between trials using image hashing'''
        return self._analyzer.analyze_video_similarity_trial()

    def get_max_dissimilarity_for_trial(self, trial_path, window):
        '''Calculate the dissimilarity within the trial given the frame sliding window.'''
        return self._analyzer.get_max_dissimilarity_for_trial(trial_path, window)

    def analyze_marker_similarity_project(self):
        '''Analyze all videos in a project and get their average rhythmicity. This assumes
        that all cam1/2 pairs are either the same or different!'''
        return self._analyzer.analyze_marker_similarity_project()

    def analyze_marker_similarity_trial(self):
        '''Analyze marker similarity for a pair of trials. Returns the mean difference for
        paired marker positions (X - X, Y - Y for each marker)'''
        return self._analyzer.analyze_marker_similarity_trial()

    @staticmethod
    def train_many_projects(parent_dir):
        '''Train and analyze multiple SDLC_XMALAB projects given a parent folder'''
        parent_path = Path(parent_dir)
        for folder in parent_path.iterdir():
            if folder.is_dir():
                project_path = folder
                deepxromm = DeepXROMM.load_project(str(project_path))
                deepxromm.train_network()
                deepxromm.analyze_videos()
