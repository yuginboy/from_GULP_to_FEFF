# from_GULP_to_FEFF
1. Convert GULP's MD simulation output *.history files to the FEFF input files, PWscf input, XYZ, XSF and for STEM files.
2. Calculate the average chi(k) spectrum [average from all snapshots]
3. Calculate the average xyz coordinates [average from all snapshots]
4. Calculate the RDF - radial distribution function [from all snapshots]
### Compare theory spectra with experimental data set:
1. Calculate the linear compositions in **_k_** and **_r_** spaces between serial snapshots [from all snapshots]
2. Calculate the R-factros (in **_k_** and **_r_** spaces)
3. Find the minimum R-factor snapshots composition (minimizing a multivariable function by using **BFGS** or **Differential evolution** methods)
4. Compare 2 and 3 models with an experiment and find the best linear combination between these models. (Ex. model A - monomer, model B - dimer, result best fit: 0.75*monomer+0.25*dimer)
5. Calculation procedures have serial and parallel (**_pathos_** library) realization
6. **_RAM_** disk could be used for accelerate calculation
7. **_numba_** library is using for a calculation speed increasing

