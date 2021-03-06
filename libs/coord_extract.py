'''
* Created by Pavlo Konstantynov
* Modified by Zhenia Syryanyy (Yevgen Syryanyy)
'''
import numpy as np
import re
import os, sys
import progressbar
# import
import shutil
from libs.classes import Unitcell
from feff.mainFEFF import feffCalcFun
from feff_calc_from_inp_parallel import FEFF_parallel_calculation_class
from feff.libs.determine_numbers_of_target_atoms import TargetAtom
from settings import ram_disk_path

limitNumOfSnapshots = 1e6

def loadCoords(file, timestep, numOfAtoms, vectForRDF, HO, numOfLinesInFile, parallel_job_numbers=5):
    '''

    :param file: .history
    :param timestep: line number of snapshots in history file
    :param numOfAtoms: number of atoms in snapshot
    :return:
    '''
    i=0
    k=0
    j=0
    atomInSnapshot = Unitcell(numOfAtoms)

    doWriteSCF = False
    doWriteXSF = False
    doWriteXYZ = True
    doWriteCFG = False
    doWriteRDF = False
    doWriteFEFFinp = False

    atomInSnapshot.rdfDist = vectForRDF
    atomInSnapshot.latCons = HO[0, 0]
    atomInSnapshot.HO = HO
    atomInSnapshot.createUniqDirOut(projectDir=os.path.dirname(file.name))
    atomInSnapshot.outNameFEFF = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameXYZ = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameXSF = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameCFG = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameRDF = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.structName = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameAverFeffInp = os.path.basename(file.name).split('.')[0]
    atomInSnapshot.outNameSCF = os.path.basename(file.name).split('.')[0]
    l = 0
    bar = progressbar.ProgressBar(maxval = len(timestep),\
                                   widgets = [progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage()])
    for line in file:
        i = i+1
        if j < len(timestep)-1:
            if i>= (timestep[0]+1) and i < (timestep[0]+4):
                # set primitive vector matrix:
                atomInSnapshot.primvec[l] = line
                l = l+1
            # Ограничения на шаг по таймстепам длинной массива (кол-вом снапшотов)
            if i >= (timestep[j]+4) and i < timestep[j+1]:
                # Обрабатываем снапшот
                i_shifted = i - (timestep[j]+4)
                if i_shifted == 0:
                    k = 0

                if k == 0:
                    tegTmp = re.findall('[a-zA-Z]+\d*|\d+', line)
                    # tegTmp = re.findall('[a-zA-Z1-9]+|\d+', line)
                    tegLine = re.findall('[a-zA-Z]+', tegTmp[0])
                    localAtomNumber = np.asarray(tegTmp[1], dtype='int').tolist()
                    atomInSnapshot.atomIndex[localAtomNumber-1] = localAtomNumber# ex: =12
                    atomInSnapshot.tag[localAtomNumber-1] = tegLine[0]#ex: =Mg


                if k == 1:
                    coordTmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)
                    coordLine = np.asarray(coordTmp, dtype = 'float')
                    atomInSnapshot.x[localAtomNumber-1] = coordLine[0]# X
                    atomInSnapshot.y[localAtomNumber-1] = coordLine[1]# Y
                    atomInSnapshot.z[localAtomNumber-1] = coordLine[2]# Z

                if k == 2:
                    k = -1

                k = k + 1

            if i == timestep[j+1]:
                # когда номер строки равен номеру строки с началом следуюшего снапшота, то увеличиваем номер
                # текущего снапшота на единицу
                if doWriteFEFFinp:
                    atomInSnapshot.writeFeffInpFileSeq()
                if doWriteSCF:
                    atomInSnapshot.writeSCFfileSeq()
                if doWriteRDF:
                    atomInSnapshot.writeRDFfileSeq()
                if doWriteXYZ:
                    atomInSnapshot.writeXYZfileSeq()

                atomInSnapshot.xAver = np.column_stack((atomInSnapshot.xAver, atomInSnapshot.x))# X
                atomInSnapshot.yAver = np.column_stack((atomInSnapshot.yAver, atomInSnapshot.y))# Y
                atomInSnapshot.zAver = np.column_stack((atomInSnapshot.zAver, atomInSnapshot.z))# Z
                if (j % 100 == 0):
                    # pass
                    if doWriteXYZ:
                        atomInSnapshot.writeXYZfileSeq()
                    if doWriteXSF:
                        atomInSnapshot.writeXSFfileSeq()
                j = j+1
                if (j % 10 == 0):
                    # print(j, ' from ', len(timestep))
                    bar.update(j)

                if j == limitNumOfSnapshots:
                    # atomInSnapshot.writeCFGfileSeq()
                    print('Snapshot Number: {0} is reached. Calculations were interrupt'.format(j))
                    break

        elif j == len(timestep)-1:
            if i >= (timestep[j]+4):
                i_shifted = i - (timestep[j]+4)
                if i_shifted == 0:
                    k = 0

                if k == 0:
                    tegTmp = re.findall('[a-zA-Z]+|\d+', line)
                    tegLine = tegTmp[0]
                    localAtomNumber = np.asarray(tegTmp[1], dtype='int').tolist()
                    atomInSnapshot.atomIndex[localAtomNumber-1] = localAtomNumber# ex: =12
                    atomInSnapshot.tag[localAtomNumber-1] = tegLine[0:2]#ex: =Mg


                if k == 1:
                    coordTmp = re.findall('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)
                    coordLine = np.asarray(coordTmp, dtype = 'float')
                    atomInSnapshot.x[localAtomNumber-1] = coordLine[0]# X
                    atomInSnapshot.y[localAtomNumber-1] = coordLine[1]# Y
                    atomInSnapshot.z[localAtomNumber-1] = coordLine[2]# Z
                if k == 2:
                    k = -1

                k = k + 1

            if i == numOfLinesInFile:
                bar.finish()
                # atomInSnapshot.writeFeffInpFileSeq()
                # atomInSnapshot.writeRDFfileSeq()
                # atomInSnapshot.writeXYZfileSeq()
                # atomInSnapshot.writeXSFfileSeq()
                if doWriteFEFFinp:
                    atomInSnapshot.writeFeffInpFileSeq()
                if doWriteSCF:
                    atomInSnapshot.writeSCFfileSeq()
                if doWriteRDF:
                    atomInSnapshot.writeRDFfileSeq()
                if doWriteXYZ:
                    atomInSnapshot.writeXYZfileSeq()

                print("output files [feff, rdf, xyz, xsf] were created for the last snapshot")

    if i < numOfLinesInFile:
        bar.finish()

    # write atoms.cfg file (the number of majorElemTag):
    atoms_cfg = TargetAtom()
    atoms_cfg.atom_type = atomInSnapshot.majorElemTag
    atoms_cfg.path_to_cfg_file = os.path.join(os.path.dirname(file.name), 'atoms.cfg')
    atoms_cfg.number_of_target_atoms = atomInSnapshot.get_num_of_major_element_tag()
    atoms_cfg.create_cfg_file()
    atoms_cfg.print_info()

    # write CFG files:
    if doWriteCFG:
        atomInSnapshot.writeCFGfileSeq()
    print("input files for QSTEM were created")
    # calculate RDF values:
    if doWriteRDF:
        atomInSnapshot.calcAndPlotMeanRDF()

    if doWriteFEFFinp:
        print('Start FEFF simulations')
        if parallel_job_numbers < 2:
            feffCalcFun(dataPath = atomInSnapshot.outDirFEFF,
                    tmpPath=atomInSnapshot.outDirFEFFtmp,
                    outDirPath=atomInSnapshot.outDirFEFFCalc,
                    plotTheData = True)
        else:
            MainObj = FEFF_parallel_calculation_class()
            # load inp dir name:
            MainObj.dirNameInp = atomInSnapshot.outDirFEFF
            # prepare vars:
            from feff.libs.dir_and_file_operations import listOfFilesFN_with_selected_ext
            MainObj.list_of_inp_files = listOfFilesFN_with_selected_ext(MainObj.dirNameInp, ext='inp')
            MainObj.dirNameOut = atomInSnapshot.outDirFEFFCalc
            # use RAM-disk:
            MainObj.is_RAM_disk_exist = True
            MainObj.path_to_RAM_disk = ram_disk_path
            # set parallel jobs number:
            MainObj.parallel_job_numbers = int(parallel_job_numbers)
            # do parallel calculation with a pathos multiprocessing tool
            MainObj.start_parallel_calculations()

    #calculation average input coordinates and stndart deviation
    atomInSnapshot.outNameAverFeffInp = os.path.basename(file.name).split('.')[0]+'_aver'
    atomInSnapshot.x = np.average(atomInSnapshot.xAver, axis=1)
    atomInSnapshot.y = np.average(atomInSnapshot.yAver, axis=1)
    atomInSnapshot.z = np.average(atomInSnapshot.zAver, axis=1)
    if doWriteFEFFinp:
        atomInSnapshot.writeAverFeffInpFile()
    shutil.rmtree(atomInSnapshot.outDirFEFFtmp)
    atoms_cfg.print_info()
    print('end of the FEFF simulations')
