# MultiLayerMapping


[![Multi-Task Learning for Scalable and Dense Multi-Layer Bayesian Map Inference](https://raw.githubusercontent.com/ganlumomo/MultiLayerMapping/master/image.png)](https://www.youtube.com/watch?v=WnFUGLBmHzc)


## Getting Started

### Building with catkin

```bash
$mkdir -p ~/catkin_ws/src
$cd ~/catkin_ws/src
catkin_ws/src$ git clone https://github.com/ganlumomo/MultiLayerMapping.git
catkin_ws/src$ cd ..
catkin_ws$ catkin_make
catkin_ws$ source ~/catkin_ws/devel/setup.bash
```

### Building using Intel C++ compiler (optional for efficiency)
```bash
catkin_ws$ source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
catkin_ws$ catkin_make -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
catkin_ws$ source ~/catkin_ws/devel/setup.bash
```

### Running the Toy Example

```bash
MultiLayerMapping/launch$ roslaunch semantics_static.launch 
```

### Running the Cassie Exp

```bash
MultiLayerMapping/launch$ roslaunch cassie_node.launch
MultiLayerMapping/rviz$ rosrun rviz rviz -d cassie.rviz
```

## Relevant Publications

If you found this code useful, please cite the following:

Multi-Task Learning for Scalable and Dense Multi-Layer Bayesian Map Inference ([PDF](https://arxiv.org/pdf/2106.14986.pdf))
```
@article{gan2021multi,
  title={Multi-Task Learning for Scalable and Dense Multi-Layer {Bayesian} Map Inference},
  author={Gan, Lu and Kim, Youngji and Grizzle, Jessy W and Walls, Jeffrey M and Kim, Ayoung and Eustice, Ryan M and Ghaffari, Maani},
  journal=IEEE_J_RO,
  note = {to appear},
  year={2022},
}
```

Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954837))
```
@ARTICLE{gan2019bayesian,
  author={L. {Gan} and R. {Zhang} and J. W. {Grizzle} and R. M. {Eustice} and M. {Ghaffari}},
  journal={IEEE Robotics and Automation Letters},
  title={Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping},
  year={2020},
  volume={5},
  number={2},
  pages={790-797},
  keywords={Mapping;semantic scene understanding;range sensing;RGB-D perception},
  doi={10.1109/LRA.2020.2965390},
  ISSN={2377-3774},
  month={April},}
```
