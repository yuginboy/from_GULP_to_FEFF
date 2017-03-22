import numpy as np
import re
import os, sys
import progressbar
# import
from libs.classes import Unitcell
from feff.mainFEFF import feffCalcFun

limitNumOfSnapshots = 10000

def loadCoords(file, timestep, numOfAtoms, vectForRDF, HO, numOfLinesInFile):
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
                    tegTmp = re.findall('[a-zA-Z]+|\d+', line)
                    # tegTmp = re.findall('[a-zA-Z1-9]+|\d+', line)
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

            if i == timestep[j+1]:
                # когда номер строки равен номеру строки с началом следуюшего снапшота, то увеличиваем номер
                # текущего снапшота на единицу
                atomInSnapshot.writeFeffInpFileSeq()
                atomInSnapshot.writeSCFfileSeq()
                atomInSnapshot.writeRDFfileSeq()
                atomInSnapshot.xAver = np.column_stack((atomInSnapshot.xAver, atomInSnapshot.x))# X
                atomInSnapshot.yAver = np.column_stack((atomInSnapshot.yAver, atomInSnapshot.y))# Y
                atomInSnapshot.zAver = np.column_stack((atomInSnapshot.zAver, atomInSnapshot.z))# Z
                if (j % 100 == 0):
                    # pass
                    atomInSnapshot.writeXYZfileSeq()
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
                atomInSnapshot.writeFeffInpFileSeq()
                atomInSnapshot.writeRDFfileSeq()
                atomInSnapshot.writeXYZfileSeq()
                atomInSnapshot.writeXSFfileSeq()
                print("output files [feff, rdf, xyz, xsf] were created for the last snapshot")

    if i < numOfLinesInFile:
        bar.finish()
    # write CFG files:
    # atomInSnapshot.writeCFGfileSeq()
    print("input files for QSTEM were created")
    # calculate RDF values:
    atomInSnapshot.calcAndPlotMeanRDF()
    print('Start FEFF simulations')
    feffCalcFun(dataPath = atomInSnapshot.outDirFEFF,
                tmpPath=atomInSnapshot.outDirFEFFtmp,
                outDirPath=atomInSnapshot.outDirFEFFCalc,
                plotTheData = True)

    #calculation average input coordinates and stndart deviation
    atomInSnapshot.outNameAverFeffInp = os.path.basename(file.name).split('.')[0]+'_aver'
    atomInSnapshot.x = np.average(atomInSnapshot.xAver, axis=1)
    atomInSnapshot.y = np.average(atomInSnapshot.yAver, axis=1)
    atomInSnapshot.z = np.average(atomInSnapshot.zAver, axis=1)
    atomInSnapshot.writeAverFeffInpFile()

    print('end of the FEFF simulations')
