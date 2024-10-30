# Contrastive Imitation Learning for Language-guided Multi-Task Robotic Manipulation
This is the official repo for [$\Sigma$-Agent](https://teleema.github.io/projects/Sigma_Agent/index.html) [CoRL 2024]

# Installation

### 1. Clone this repo
`git clone https://github.com/TeleeMa/Sigma-Agent.git`
### 2. Install conda environment and simulator
Following the [RVT](https://github.com/NVlabs/RVT) to install conda environment and CoppeliaSim.
### 3. Install RVT, PyRep, RLBench, YARR and PerAct Colab
```
cd Sigma-Agent
pip install -e .
pip install -e rvt/libs/PyRep 
pip install -e rvt/libs/RLBench 
pip install -e rvt/libs/YARR 
pip install -e rvt/libs/peract_colab
``` 



## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@article{ma2024contrastive,
&nbsp;title={Contrastive Imitation Learning for Language-guided Multi-Task Robotic Manipulation},
&nbsp;author={Ma, Teli and Zhou, Jiaming and Wang, Zifan and Qiu, Ronghe and Liang, Junwei},
&nbsp;journal={arXiv preprint arXiv:2406.09738},
&nbsp;year={2024}
} </pre>