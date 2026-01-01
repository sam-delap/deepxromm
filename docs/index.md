# Welcome to DeepXROMM!

DeepXROMM integrates DeepLabCut's neural network tracking capabilities with XMAlab's XROMM (X-ray Reconstruction of Moving Morphology) workflow, enabling automated marker tracking for X-ray motion analysis.

## Key Features

- **XMAlab Integration**: Seamlessly converts between XMAlab and DeepLabCut data formats
- **Flexible Tracking Modes**: Choose from 2D, per-camera, or RGB merged tracking strategies
- **Automated Batch Training**: Train multiple projects with a single command
- **Retraining Workflow**: Iteratively improve networks by identifying and correcting outlier frames
- **XMAlab-Style Autocorrection**: Apply image processing techniques for marker refinement
- **Video Analysis Tools**: Compare trials and validate tracking quality

## Quick Start

```
┌─────────────────┐
│  1. Install     │──> Follow the Installation guide
└────────┬────────┘
         │
┌────────▼────────┐
│  2. Create      │──> Create project and import XMAlab data
│     Project     │
└────────┬────────┘
         │
┌────────▼────────┐
│  3. Convert &   │──> xma_to_dlc() + create_training_dataset()
│     Prepare     │
└────────┬────────┘
         │
┌────────▼────────┐
│  4. Train       │──> train_network()
│     Network     │
└────────┬────────┘
         │
┌────────▼────────┐
│  5. Analyze     │──> analyze_videos() + dlc_to_xma()
│     Videos      │
└────────┬────────┘
         │
┌────────▼────────┐
│  6. Import to   │──> Import predictions into XMAlab
│     XMAlab      │
└─────────────────┘
```

### Detailed Steps

1. **[Install](install.md)** DeepXROMM and its dependencies
2. **[Create a project](usage.md#getting-started-and-creating-a-new-project)** and set up your directory structure
3. **[Import your data](usage.md#importing-your-data-and-loading-the-project)** from XMAlab
4. **[Train your network](usage.md#training-the-project)** on tracked markers
5. **[Analyze videos](usage.md#using-a-trained-network-to-track-your-trials)** with the trained model
6. **[Import results](usage.md#converting-dlc-predictions-back-to-xmalab-format)** back into XMAlab

## Tracking Modes

DeepXROMM supports three tracking modes to suit different experimental setups:

- **2D Mode** (default): Combines both camera views into a single DeepLabCut project with shared bodypart names. Both cameras train together in one network.
  - **Best for:** Standard stereo X-ray setups where both cameras view the same subject

- **Per-Camera Mode**: Creates separate DeepLabCut projects and networks for cam1 and cam2. Each camera trains independently.
  - **Best for:** Different camera angles requiring specialized training, or when cameras have very different characteristics

- **RGB Mode**: Merges cam1 and cam2 into a single RGB video (cam1 → red channel, cam2 → green channel). Bodyparts are labeled with camera suffixes.
  - **Best for:** Leveraging spatial relationships between camera views, advanced training scenarios

See the [Mode parameter documentation](config.md#neural-network-customization) for detailed information about each mode.

## Documentation Structure

- **[Installation Guide](install.md)**: Set up DeepXROMM and dependencies
- **[Usage Guide](usage.md)**: Complete workflow from project creation to analysis
- **[Config File Reference](config.md)**: Detailed explanation of all configuration parameters
- **[Contributing Guide](CONTRIBUTING.md)**: Guidelines for contributing to the project

## Getting Help

If you encounter issues or have questions:
- Check the [Usage Guide](usage.md) for detailed workflow instructions
- Review the [Config File Reference](config.md) for parameter descriptions
- Consult the [Contributing Guide](CONTRIBUTING.md) for development setup

## Project Layout

After creating a project, your directory structure will look like this:

    your-project/
    │   project_config.yaml       # Main configuration file
    │
    ├───your-project-SD-YYYY-MM-DD/   # DeepLabCut project folder
    ├───trainingdata/             # Trials with tracked markers for training
    └───trials/                   # Trials to analyze with trained network
