# Flower
Flower is a new visualization tool for in-depth study of multi-sensor recordings in the time domain. It has been released for public download as a fully functioning tool available for experimental, research, and creative use. Flower uses unsupervised machine learning to extract latent representations for time-series data (EEG in particular) and show them through different visualization settings. In particular, it adds color and thickness to time-series plots, making them easier to understand and compare. Flower aims to enable more natural intuition around data results, using machine intelligence to translate time-series data for improved understanding by the human eye.
## Installation
1. Download and Install Anaconda from: https://www.anaconda.com/distribution/
2. Clone the repository in a directory:
```console
git clone https://github.com/NeoVand/Flower2.git
```
3. Create a python 3.7 environment by typing this command in the Anaconda Prompt (or any shell with conda in PATH):
```console
cd Flower2
conda env create --name mneflower --file environment.yml
```
4. Activate the environment:
```console
conda activate mneflower
```
if you are using macOS then run this line:
```console
pip install "PyQt5>=5.10"
```
5. Run the application:
```console
python app.py
```
6. Load the UI:
go to local host with port 5000 by pointing chrome to `127.0.0.1:5000`