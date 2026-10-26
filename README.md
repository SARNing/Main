1 Background
Synthetic Aperture Radar (SAR) is an active microwave remote sensing technology that provides high-resolution Earth observation under all-weather and day/night conditions. Unlike optical remote sensing, SAR is independent of solar illumination and is largely unaffected by clouds, rain, fog, and other atmospheric obstructions. These properties make SAR highly valuable for military target recognition, maritime surveillance, disaster assessment, urban infrastructure monitoring, and many other applications.
However, SAR imagery also introduces distinctive challenges. SAR images are affected by multiplicative speckle noise, strong azimuth sensitivity, geometric distortions such as layover and foreshortening, and significant appearance variations caused by changes in viewing geometry. As a result, the same target class may exhibit large intra-class variation across different imaging conditions, while different target classes may share similar local scattering responses. These factors make stable and discriminative feature learning difficult, especially when labeled samples are scarce.
In real-world SAR applications, high-resolution, multi-condition, expert-annotated target samples are expensive and time-consuming to acquire. Rare or mission-critical target categories often have only a few labeled examples, leading to severe class imbalance and limited training data. Conventional deep learning models are data-hungry and may overfit under such conditions, resulting in degraded generalization and reduced operational robustness.
Few-shot learning (FSL) offers a promising paradigm for learning transferable representations from only a small number of labeled examples. Existing FSL methods can be broadly categorized into meta-learning, metric-learning, and data-augmentation/generative approaches. Unlike conventional few-shot classification, which focuses on recognizing unseen categories, this work investigates few-shot cross-view SAR target recognition. In this setting, support and query samples belong to the same target categories but are collected from mutually exclusive azimuth intervals, and possibly from different depression angles. The model must therefore maintain discriminative representations under limited azimuth annotations and significant viewpoint variations.

2 Environment Setup
conda create -n myenv python=3.10

conda activate myenv

conda install numpy 

conda install pytorch torchvision torchaudio cpuonly -c pytorch

3 Run the Code
Before training and few-shot evaluation, the MSTAR dataset must be divided according to the experimental protocol. In this work, support and query samples are selected from mutually exclusive azimuth intervals. Therefore, please first run the dataset splitting script to generate the required training, support, and query subsets：
python jpg_split.py
then:
python runner.py

4 Dataset Download
https://pan.baidu.com/s/1oFkseSGO81tABbqC4ZoVVg
    └── MSTAR/
        ├── 17deg/
        │   ├── 2S1/
        │   ├── BMP2/
        │   ├── BRDM2/
        │   ├── BTR60/
        │   ├── BTR70/
        │   ├── D7/
        │   ├── T62/
        │   ├── T72/
        │   ├── ZIL131/
        │   └── ZSU-23-4/
        └── 15deg/
            ├── 2S1/
            ├── BMP2/
            ├── BRDM2/
            ├── BTR60/
            ├── BTR70/
            ├── D7/
            ├── T62/
            ├── T72/
            ├── ZIL131/
            └── ZSU-23-4/
