# ECGTokenizer
This software is a modified version based on
[Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model](https://github.com/PKUDigitalHealth/HeartLang)  
originally created by Jiarui Jin and Haoyu Wang.

# Main Changes
- Modified the repository to enable distribution via GitHub:
```
pip install git+https://github.com/SuperHotDogCat/ECGTokenizer.git
```

- Created `class ECGTokenizer` to handle everything from QRS segmentation to ID assignment in a single process:
```
from ecgtokenizer import ECGTokenizer
```
