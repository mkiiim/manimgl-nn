
## Manim and Environment Issues
  
Key requirements:
- miniforge 3.24.0 (python 3.10 default)
- ManimGL v1.6.1
  
NOTE - Had to reinstall miniforge 3.24.0 in order to solve environment and dependency issues including:
- `python` version issues (suggested use of `python==3.10` which is default for miniforge `3.24.0` )
- `manimpango` and `cairo.h` issues
- numpy issues (suggested version 1.24 to resolve [ValueError: operands could not be broadcast together with shapes (24,3) (0,3)](https://github.com/3b1b/manim/issues/2053#top))
  
### Create miniforge conda environment

```zsh
conda create --name manimgl --clone base
conda activate manimgl
pip install manimgl
pip install numpy==1.24
```