
# from_GULP_to_FEFF ([GNU GPLv3](http://gplv3.fsf.org/))
---
### Automatic EXAFS [FEFF](http://monalisa.phys.washington.edu/feffproject-feff.html)  simulation and different file formats conversion 
1. Convert [**GULP**'s](https://gulp.curtin.edu.au/gulp/overview.cfm) **MD** simulation output *.history files to the **FEFF** input files, [**PWscf**](http://www.quantum-espresso.org/) input, **XYZ**, **XSF** and for **STEM** files.
1. automatic EXAFS **FEFF** calculations for all created **FEFF** input files (_serial and parallel realization_)
1. Calculate the average **_chi(k)_** spectrum [_average from all snapshots_]
1. Calculate the average **_xyz_** coordinates [_average from all snapshots_]
1. Calculate the **RDF** - radial distribution function [_from all snapshots_]

---
### Compare theory spectra with experimental data set:

1. Calculate the linear compositions in **_k_** and/or **_r_** spaces between serial snapshots [_from all snapshots_]
1. Calculate the **R**-factros (in **_k_** and **_r_** spaces)
1. Find the minimum **R**-factor snapshots composition (minimizing a multivariable function by using **BFGS** or **Differential evolution** methods)
1. Compare 2 and 3 models with an experiment and find the best linear combination between these models. (Ex. for interpretation EXAFS spectra for bulk material we used two models fit, model **A** - _monomer_, model **B** - _dimer_, and got the best fit result: **0.75** * _monomer_ + **0.25** *  _dimer_, which could be iterpreted like: in bulk material we have 75% of _monomer_'s phase and 25% of _dimer_'s phase)
1. Calculation procedures have serial and parallel ([**_pathos_**](https://pypi.python.org/pypi/pathos) library) realization
1. **_RAM_** disk could be used for accelerate calculation (_user choice_)
1. [**_numba_**](https://numba.pydata.org/) library is using for a calculation speed increasing

---
### Authors
1. **Yevgen Syryanyy** [_main contributor_]
1. Pavlo Konstantynov
