conda create -n myenv python=3.10

conda activate myenv

conda install pytorch torchvision torchaudio cpuonly -c pytorch

python runner.py
