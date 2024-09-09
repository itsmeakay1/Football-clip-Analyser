# Football Clip Analysis Pipeline

This project provides a robust pipeline for analyzing football clips using computer vision techniques. It utilizes YOLOv8 for object detection and tracking, allowing users to process football video clips and obtain analyzed output videos.

## Features

- Automated analysis of football video clips
- Object detection and tracking using YOLOv8
- Easy-to-use command-line interface
- Customizable input and output directories

## Requirements

- Python 3.7+
- ultralytics
- supervision
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn
- ipykernel

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/football-clip-analysis.git
   cd football-clip-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your input football video clips in the `input` folder.

2. Run the analysis script:
   ```
   python main.py
   ```

3. Find the analyzed videos in the `output_video` folder.

## Example Clips

### Before Analysis



https://github.com/user-attachments/assets/e2adb662-4ea3-4acf-8180-dfd205457a02



### After Analysi



https://github.com/user-attachments/assets/44f4b7b5-686b-470b-9ce7-aa67e79186e2



## Customization

You can modify the `main.py` script to adjust detection parameters, tracking settings, or add additional analysis features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project uses YOLOv8 for object detection and tracking. We thank the Ultralytics team for their excellent work.
