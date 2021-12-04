# PINN-AM

A physics-informed neural network for application in additive manufacturing problem. The following article outline our code:
```
@article{zhu2021machine,
  title={Machine learning for metal additive manufacturing: predicting temperature and melt pool fluid dynamics using physics-informed neural networks},
  author={Zhu, Qiming and Liu, Zeliang and Yan, Jinhui},
  journal={Computational Mechanics},
  volume={67},
  number={2},
  pages={619--635},
  year={2021},
  publisher={Springer}
}

```

## Running the tests
1D demo case is included in the current code. Please check the details in our paper. The meaning of folders are listed as followed:

```
dat          for FEM solution of 1D problem
resolution   for PINN with different number of neurons
soft_hard    for PINN with soft and hard enforcement of boundary conditions
```
## Authors

Qiming Zhu, phd student in UIUC,          qiming2@illinois.edu  
Jinhui Yan, Assistant Professor in UIUC,  yjh@illinois.edu  

Feel free to send me email, if you have problems with using this code.  


