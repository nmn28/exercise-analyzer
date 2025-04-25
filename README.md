# Exercise Form Analyzer

A Python application that analyzes exercise form using computer vision and pose estimation. The project primarily uses MediaPipe for pose estimation (in the latest `main.py` implementation), while also maintaining support for YOLOv8 as an alternative backend.

## Features

- **Primary MediaPipe implementation** (`main.py`):
  - Robust person tracking and detection
  - Temporal smoothing for stable analysis
  - Efficient CPU-based processing
  - Detailed form feedback and scoring
  
- **Alternative YOLOv8 implementation** (`exercise_analyzer.py`):
  - Higher accuracy for complex movements
  - Multiple model size options for speed/accuracy tradeoffs
  - GPU acceleration support
  
- **Rep detection and counting** with automatic identification of exercise phases

- **Detailed form analysis** including:
  - Squat depth measurement
  - Hip hinge analysis
  - Back angle and stability tracking
  - Joint angle calculations

- **Comprehensive feedback** with specific deductions and improvement suggestions

- **Visual output** including:
  - Pose skeleton visualization
  - Real-time score display
  - Rep counter
  - Form feedback
  - Rep status (IN REP/REST)
  - Joint angle measurements

## Installation

1. Ensure you have Python 3.11 or newer installed.
2. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Clone this repository and navigate to it:
   ```bash
   git clone [repository-url]
   cd exercise-analyzer
   ```
4. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

### Primary MediaPipe Implementation (main.py)

The recommended implementation using MediaPipe with robust person tracking:

```bash
poetry run python main.py [video_path] [options]
```

#### Arguments
- **video_path**: Path to the input video file (required)

#### Options
- `--min-detection-confidence`: Minimum confidence for pose detection (default: 0.7)
- `--min-tracking-confidence`: Minimum confidence for pose tracking (default: 0.7)

#### Example
```bash
poetry run python main.py path/to/video.mp4 --min-detection-confidence 0.8
```

### Alternative YOLOv8 Implementation (exercise_analyzer.py)

For cases where higher accuracy is needed and GPU resources are available:

```bash
poetry run python exercise_analyzer.py [exercise] [video_path] [options]
```

#### Arguments
- **exercise**: Type of exercise to analyze (required). Options: `squats`, `dumbbell_shoulder_press`
- **video_path**: Path to the input video file (required)

#### Options
- `--model-size`: YOLOv8 model size (optional, default: `x`):
  - `n`: Nano (fastest, least accurate)
  - `s`: Small
  - `m`: Medium (balanced)
  - `l`: Large
  - `x`: Extra Large (slowest, most accurate)

#### Example
```bash
poetry run python exercise_analyzer.py squats path/to/video.mp4 --model-size m
```

## Output

The script generates:

1. An annotated video file with:
   - Pose skeleton
   - Rep counter
   - Current score
   - Form feedback
   - Rep status indicator
   - Joint angle measurements

2. Console output showing:
   - Number of reps detected
   - Score for each individual rep
   - Specific form deductions
   - Overall average score

## Implementation Comparison

### MediaPipe (main.py) - Primary Implementation
- **Advantages**: 
  - Faster processing, works well on CPU
  - More robust person tracking
  - Stable temporal smoothing
  - Better for real-time applications
- **Best for**: Most use cases, especially when GPU resources are limited

### YOLOv8 (exercise_analyzer.py) - Alternative Implementation
- **Advantages**:
  - Higher accuracy for complex movements
  - Configurable model sizes for different performance needs
  - Better for detailed analysis when processing time isn't critical
- **Best for**: Offline analysis where maximum accuracy is required

## Requirements

- Python ≥ 3.11
- Poetry
- Sufficient disk space for models
- Webcam or video file for analysis

## Notes

- The first run will download the required models.
- Processing speed depends on your hardware and chosen implementation.
- For best results:
  - Ensure the subject is fully visible in the video.
  - Wearing fitting clothes can improve pose estimation accuracy.
  - Record from a side angle for better joint visibility.
