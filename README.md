# nnGuSTO

This is the Julia code for [Learning-based Warm-Starting for Fast Sequential Convex Programming and Trajectory Optimization](http://asl.stanford.edu/wp-content/papercite-data/pdf/Banerjee.Lew.Bonalli.ea.AeroConf20.pdf), in which a neural network is used to warm-start a trajectory optimization algorithm ([GuSTO](https://arxiv.org/abs/1903.00155)) leading to faster convergence.

# Requirements
All required packages can be installed by cloning this repository, navigating to the project directory, and using the following command in the Julia REPL package manager:
```
(v1.0) pkg> activate .

(nnGuSTO) pkg> instantiate
```

# Quickstart
An example notebook can be run through:
```
jupyter notebook demo_gusto_julia.ipynb 
```
More detailed analysis can be found in the following notebooks:
```
jupyter notebook src/notebooks/
```

## References
* S. Banerjee, T. Lew, R. Bonalli, A. Alfaadhel, I. A. Alomar, H. M. Shageer, and M. Pavone, [“Learning-based Warm-Starting for Fast Sequential Convex Programming and Trajectory Optimization,”](http://asl.stanford.edu/wp-content/papercite-data/pdf/Banerjee.Lew.Bonalli.ea.AeroConf20.pdf) in IEEE Aerospace Conference, Big Sky, Montana, 2020.
