# super-mario-bros-RL
Playing Super Mario Bros in Atari environment with RL Algorithms 

## Prerequisites
 - This project was developed with Python 3.11.9, but any version between 3.9.x and 3.11.x should work
 - Follow [PyTorch's](https://pytorch.org/get-started/locally/) instructions to set up the PyTorch library on your machine, as the steps are different according to what CUDA version you are using, if any
 - Install the exact requirements using `pip install -r requirements.txt`

## Necessary Dependencies 
```pip install numpy```

```pip install cv2```

```pip install torch==2.5.1```

```pip install nes_py==8.2.1```

```pip install gym==0.25.1```

```pip install gym_super_mario_bros==7.4.0```

```gymnasium==0.28.1```

## Necessary System Dependency
```conda install conda-forge::libglu``` # ImportError: Library "GLU" not found.

```conda install -c conda-forge libglu```

### For GPU
```nvidia-smi```

GPU Name: NVIDIA A100 80GB PCIe 

CUDA version: 12.4

```pip show torch``` to check torch version


## How to Run
To Train, ```python main.py --train```

To test, ```python main.py --test``` or ```python main.py --test --record```