# HATS-PyTorch

A *unofficial* PyTorch+CUDA implementation of "HATS: Histograms of Averaged Time 
Surfaces for Robust Event-based Object Classification", Sironi et al. 

Tested on Pytorch 1.6.0, CUDA 10.1

### Installation

- Clone this repository (add `--recursive` if you want to run the demo script)<br>
`git clone https://github.com/marcocannici/hats_pytorch.git`
- Build and install the CUDA kernels:
`cd cuda; python setup.py --install`
- Install the `hats_pytorch` package:
`python setup.py --install`

You can test the implementation on the 
[N-Cars](https://www.prophesee.ai/2018/03/13/dataset-n-cars/) dataset by running 
the following command (you must have cloned the repository with `--recursive` 
for this to work):

`python demo/run.py --data_dir /path/to/ncars/train --batch_size 1`

Time to extract representations from all the N-Cars training samples on a 
GeForce GTX 1080ti is 01:27 (3.925 ms/sample) with batch_size 1, and 00:47 
(0.758 ms/sample) with batch_size 64


### Usage

```python
from hats_pytorch import HATS
hats = HATS((100, 120), r=3, k=10, tau=1e9, delta_t=100000, fold=True)
hats.to('cuda:0')
histograms = hats(events, lengths)
```


### Disclaimer

This is unofficial code and, as such, the implementation may differ from the one
reported in the paper. If you find any error or difference with the paper, do not
hesitate to report it! :smiley:

