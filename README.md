# from_GULP_to_FEFF
1. Convert GULP's MD simulation output *.history files to the FEFF input files, PWscf input, XYZ, XSF and for STEM files.
2. Calculate the average chi(k) spectrum [average from all snapshots]
3. Calculate the average xyz coordinates [average from all snapshots]
4. Calculate the RDF - radial distribution function [from all snapshots]
### Compare theory spectra with experimental data set:
1. Calculate the linear compositions in k and r spaces between serial snapshots [from all snapshots]
2. Calculate the R-factros
3. Find the minimum R-factor snapshots composition
4. Compare 2 and 3 models with experiment and find best linear combination between these models. (Ex. model A - monomer, model B - dimer, result best fit: 0.75*monomer+0.25*dimer)


