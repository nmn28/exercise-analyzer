# Exercise Form Analyzer

A Python script that analyzes exercise form using YOLOv8 pose estimation. The script processes videos of exercises (currently supporting squats and shoulder presses), identifies individual reps, and provides detailed form feedback and scoring.

## Features

- **Real-time pose estimation** using YOLOv8
- **Rep detection and counting**
- **Individual rep scoring**
- **Form feedback**
- **Temporal smoothing** for stable tracking
- Customizable **model sizes** for different speed/accuracy trade-offs
- **Progress bar** for processing status
- **Output video** with overlays, including:
  - Pose skeleton
  - Current score
  - Rep counter
  - Form feedback
  - Rep status (IN REP/REST)

## Installation

1. Ensure you have Python 3.8 or newer installed.
2. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Clone this repository and navigate to it:
   ```bash
   git clone [repository-url]
   cd exercise-form-analyzer
   ```
4. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

### Basic Usage

```bash
poetry run python exercise_analyzer.py [exercise] [video_path] [options]
```

### Arguments

- **exercise**: Type of exercise to analyze (required). Options: `squats`, `dumbbell_shoulder_press`
- **video_path**: Path to the input video file (required). Supports formats such as `.mp4` and `.mov`.

### Options

- `--model-size`: YOLOv8 model size (optional, default: `x`):
  - `n`: Nano (fastest, least accurate)
  - `s`: Small
  - `m`: Medium (balanced)
  - `l`: Large
  - `x`: Extra Large (slowest, most accurate)

### Examples

- Analyze squat form using the default (extra large) model:
  ```bash
  poetry run python main.py squats path/to/video.mov
  ```

- Analyze shoulder press using the nano model for faster processing:
  ```bash
  poetry run python main.py dumbbell_shoulder_press path/to/video.mp4 --model-size n
  ```

- Use the medium model for balanced speed and accuracy:
  ```bash
  poetry run python main.py squats path/to/video.mov --model-size m
  ```

## Output

The script generates:

1. An annotated video file named `analyzed_[exercise]_yolov8[model-size].mp4`
2. Console output showing:
   - Number of reps detected
   - Score for each individual rep
   - Overall average score

## Model Size Comparison

| Model Size | File Size | Description |
|------------|-----------|-------------|
| `n` (Nano) | ~35MB     | Fastest, suitable for real-time applications |
| `s` (Small) | ~68MB    | Good balance for faster machines |
| `m` (Medium) | ~98MB   | Recommended for general use |
| `l` (Large) | ~165MB   | High accuracy, slower |
| `x` (Extra Large) | ~200MB | Highest accuracy, slowest |

### Choosing the Right Model Size

- Use `n` or `s` for real-time or quick analysis.
- Use `m` for balanced performance.
- Use `l` or `x` for detailed analysis when speed is less critical.

## Requirements

- Python ≥ 3.8
- Poetry
- Sufficient disk space for YOLOv8 models
- Webcam or video file for analysis

## Notes

- The first run will download the selected YOLOv8 model.
- Processing speed depends on your hardware and chosen model size.
- For best results:
  - Ensure the subject is fully visible in the video.
  - Wearing fitting clothes can improve pose estimation accuracy.

