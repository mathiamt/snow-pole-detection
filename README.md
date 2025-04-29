# Snow Pole Detection Project

## Setup Instructions

1. Clone this repository
2. Create a virtual environment and install dependencies:
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    pip install -r requirements.txt
3. Clone YOLOv7:
    mkdir -p dependencies
    git clone https://github.com/WongKinYiu/yolov7.git dependencies/yolov7
4. Install YOLOv7 requirements:
    pip install -r dependencies/yolov7/requirements.txt
    