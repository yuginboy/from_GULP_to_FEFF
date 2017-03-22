import numpy as np
import periodictable as pd
from libs.dir_and_file_operations import createUniqFile
from libs.inputheader import writeHeaderToInpFile
from libs.dir_and_file_operations import create_out_data_folder
import os
from libs.average_rdf import AverageRDF
def angstrom_to_bohr(x=0):
    return x*1.8897259885789 # 1 angstrom [Å] = 1.8897259885789 Bohr radius [b, a.u.]
def bohr_to_angstrom(x=0):
    return x/1.8897259885789 # 1 angstrom [Å] = 1.8897259885789 Bohr radius [b, a.u.]

class Unitcell(AverageRDF):
    def __init__(self, numOfAtoms=1):
        # lattice constant value:
        self.latCons = 1

        # components of lattice vector:
        self.HO = np.zeros((3, 3))
        self.majorElemTag = 'Mn'
        self.l_scmt = 2
        self.l_fms = 2
        self.stoichiometry = 4
        self.x = np.zeros(numOfAtoms)
        self.xAver = np.zeros(numOfAtoms)
        self.y = np.zeros(numOfAtoms)
        self.yAver = np.zeros(numOfAtoms)
        self.z = np.zeros(numOfAtoms)
        self.zAver = np.zeros(numOfAtoms)
        self.atomIndex = np.zeros(numOfAtoms)
        # self.atomIndex = np.arange(0, numOfAtoms, 1)
        self.tag = np.zeros(numOfAtoms, dtype='U2')
        self.tag[:] = 'H'
        self.r = np.zeros(numOfAtoms)
        self.ipot = np.zeros(numOfAtoms, dtype='int')
        self.outDirFEFF =        'test'
        self.outDirXYZ =         'test'
        self.outDirRDF =         'test'
        self.outDirXSF =         'test'
        self.outDirCFG =         'test'
        self.outDirAverFeffInp = 'test'
        self.outDirSCF =         'test'
        self.scfPseudoDirPath = '/home/yugin/installed/qe-6.1/pseudo/'

        # file name:
        self.outNameFEFF = 'feff'
        self.outNameXYZ = 'struct'
        self.outNameRDF = 'rdf'
        self.outNameXSF = 'struct'
        self.outNameCFG = 'stem'
        self.outNameAverFeffInp = 'aver_inp'
        self.outNameSCF = 'scf_in'

        # rdfDist is vector of distances for radial distribution calculation:
        self.rdfDist = np.linspace(0, 6, 4)
        # three-line container for PRIMVEC:
        self.primvec = []
        self.primvec.append('   1.000       0.000       0.000\n')
        self.primvec.append('   0.000       1.000       0.000\n')
        self.primvec.append('   0.000       0.000       1.000\n')

        super().__init__()

    def shiftZeroToCenterOfStruct(self):

        halfLatCons = self.latCons / 2
        for i in range(len(self.x)):
            if self.x[i] > halfLatCons:
                self.x[i] = self.x[i] - self.latCons
            elif self.x[i] < - halfLatCons:
                self.x[i] = self.x[i] + self.latCons

            if self.y[i] > halfLatCons:
                self.y[i] = self.y[i] - self.latCons
            elif self.y[i] < - halfLatCons:
                self.y[i] = self.y[i] + self.latCons

            if self.z[i] > halfLatCons:
                self.z[i] = self.z[i] - self.latCons
            elif self.z[i] < - halfLatCons:
                self.z[i] = self.z[i] + self.latCons

    def distance(self, x=0, y=0, z=0):
        self.shiftZeroToCenterOfStruct()
        # distance calculation betwenn i-th atoms and selected position(x,y,z)
        for i in range(0, len(self.x), 1):
            self.r[i] = np.sqrt((x - self.x[i]) ** 2 +
                                (y - self.y[i]) ** 2 +
                                (z - self.z[i]) ** 2)

    def lineOutStr(self, index=0):
        # line formation for comfort printing to console
        for i in range(0, len(self.x), 1):
            print('x = {0:.5f}, y = {1:.5f}, z = {2:.5f}, ipot = {3}, tag = {4}, distance = {5:.5f}, index = {6}, '
                  .format(self.x[i], self.y[i], self.z[i], self.ipot[i], self.tag[i], self.r[i], self.atomIndex[i] + 1))

    def writeTableToFile(self,
                         filename='test.txt',
                         overwriteOrNot=True):
        # writing table of coordinates and attributes of atoms to file
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')

        header0 = 'ATOMS                          * this list contains {0} atoms\n'.format(len(self.x))
        header1 = ' *   x          y          z          ipot  tag        distance\n'
        file.write(header0)
        file.write(header1)
        for i in range(0, len(self.x), 1):
            line = ' {0:+.5f}    {1:+.5f}    {2:+.5f}     {3}     {4}         {5:.5f}   {6:.0f}\n' \
                .format(self.x[i], self.y[i], self.z[i], self.ipot[i], self.tag[i], self.r[i],
                        self.atomIndex[i]).replace('+', ' ')
            file.write(line)
        file.write('END')
        file.close()

    def writeTableToXYZfile(self,
                            filename='test.xyz',
                            overwriteOrNot=True):
        # writing table of coordinates and attributes of atoms to file
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')

        header0 = '{0}\n\n'.format(len(self.x))
        file.write(header0)
        for i in range(0, len(self.x), 1):
            line = '{0}\t{1:+.5f}\t{2:+.5f}\t{3:+.5f}\n' \
                .format(self.tag[i], self.x[i], self.y[i], self.z[i]).replace('+', ' ')
            file.write(line)
        file.close()

    def write_relative_xyz_TableTofile(self,
                            filename='test.xyz',
                            overwriteOrNot=True):
        # writing table of relative coordinates and attributes of atoms to file
        # prepare table for scf.in file
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')

        for i in range(0, len(self.x), 1):
            line = '{0}\t{1:+.9f}\t{2:+.9f}\t{3:+.9f}\n' \
                .format(self.tag[i], self.x[i] / self.HO[0, 0], self.y[i] / self.HO[1, 1],
                        self.z[i] / self.HO[2, 2]).replace('+', ' ')
            file.write(line)
        file.close()

    def create_SCF_header(self, prefix='tmp_scf'):
        # prepare header for scf.in file
        uniqTags, uniqTagsCounts = np.unique(self.tag, return_counts=True)
        txt = ''
        # control part:
        txt = txt + '&control\n'
        txt = txt + '  calculation = \'scf\'\n'
        txt = txt + '  restart_mode=\'from_scratch\'\n'
        txt = txt + '  prefix=\'{0}\'\n'.format(prefix)
        txt = txt + '  pseudo_dir=\'{0}\'\n'.format(self.scfPseudoDirPath)
        txt = txt + '  outdir=\'./\'\n'
        txt = txt + '  verbosity=\'high\'\n'
        txt = txt + '  max_seconds=165600\n'
        txt = txt + '/\n'

        # system part:
        txt = txt + '&system\n'
        txt = txt + '  ibrav=1,\n'
        txt = txt + '  nat={0},\n'.format(len(self.x))
        txt = txt + '  ntyp={0},\n'.format(len(uniqTags))
        txt = txt + '  celldm(1)={0:9f}\n'.format(angstrom_to_bohr(self.HO[0, 0]))
        txt = txt + '  ecutwfc=30\n'
        txt = txt + '  ecutrho=300\n'
        txt = txt + '  occupations=\'smearing\', smearing=\'gaussian\', degauss=0.0025D0\n'
        txt = txt + '  nspin=2\n'
        txt = txt + '  starting_magnetization(1)=0.8\n'
        txt = txt + '  tot_charge=0.0\n'
        txt = txt + '/\n'

        # electrons part:
        txt = txt + '&electrons\n'
        txt = txt + '  electron_maxstep=100,\n'
        txt = txt + '  diagonalization=\'david\',\n'
        txt = txt + '  conv_thr =  1.0d-06,\n'
        txt = txt + '  mixing_beta = 0.4,\n'
        txt = txt + '/\n'

        # ions part:
        txt = txt + '&ions\n'
        txt = txt + '  ion_dynamics=\'bfgs\'\n'
        txt = txt + '/\n'

        # cell part:
        txt = txt + '&cell\n'
        txt = txt + '  cell_dynamics=\'bfgs\'\n'
        txt = txt + '  cell_dofree=\'xyz\'\n'
        txt = txt + '/\n'

        # ATOMIC_SPECIES part:
        txt = txt + 'ATOMIC_SPECIES\n'
        txt = txt + '  Mn  54.938   Mn.pbe-sp-van_mit.UPF\n'
        txt = txt + '  Ga  69.723   Ga.pbe-n-van.UPF\n'
        txt = txt + '  As  74.9216  As.pbe-n-van.UPF\n'
        txt = txt + '\n'

        # K_POINTS part:
        txt = txt + 'K_POINTS { automatic }\n'
        txt = txt + '2 2 2 1 1 1\n'
        txt = txt + '\n'


        # if CELL_PARAMETERS is not commented then we obtain error:
        # Error in routine cell_base_init (2):
        # redundant data for cell parameters
        # # CELL_PARAMETERS part:
        # txt = txt + 'CELL_PARAMETERS (alat= {0:9f})\n'.format(angstrom_to_bohr(self.HO[0, 0]))
        # txt = txt + '   1.000000000   0.000000000   0.000000000\n'
        # txt = txt + '   0.000000000   1.000000000   0.000000000\n'
        # txt = txt + '   0.000000000   0.000000000   1.000000000\n'
        # txt = txt + '\n'

        # ATOMIC_POSITIONS part:
        txt = txt + 'ATOMIC_POSITIONS (crystal)\n'

        return txt

    def writeSCFtoFile (self, filename='test.in', overwriteOrNot=True):
        # write header and relative coordinates table to the file:
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')
        prefix = os.path.basename(filename).split('.')[0]
        header = self.create_SCF_header(prefix=prefix)
        # write header:
        file.write(header)
        file.close()
        # write relative coordinates table:
        self.write_relative_xyz_TableTofile(filename=filename, overwriteOrNot=False)


    def writeSCFfileSeq(self, ext='in'):
        # write SCF file in Seq mode:
        mask = self.outNameSCF + '_'
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        folder = self.outDirSCF
        filename = os.path.join(folder, 'test.' + ext)
        self.backUpStruct()
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            self.writeSCFtoFile(filename=fn, overwriteOrNot=True)
            self.restoreFromBackUp()



    def centrShift(self, centrAtomIndex=1):
        # Shift atoms in orogon structure relative choosen position (x,y,z)
        self.x = self.x - self.x[centrAtomIndex]
        self.y = self.y - self.y[centrAtomIndex]
        self.z = self.z - self.z[centrAtomIndex]
        self.distance(x=self.x[centrAtomIndex],
                      y=self.y[centrAtomIndex],
                      z=self.z[centrAtomIndex])
        self.sortByDistance()

    def sortByDistance(self):
        # Sort arrays by distance
        sortedInd = np.argsort(self.r)
        self.r = self.r[sortedInd]
        self.x = self.x[sortedInd]
        self.y = self.y[sortedInd]
        self.z = self.z[sortedInd]
        self.ipot = self.ipot[sortedInd]
        self.tag = self.tag[sortedInd]
        self.atomIndex = self.atomIndex[sortedInd]

    def ipotTableConstructor(self,
                             filename='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test\\test.txt',
                             overwriteOrNot=True):
        # Search all uniqeu atom tags and calc the number of times each of the unique elements comes up in the array
        uniqTags, uniqTagsCounts = np.unique(self.tag, return_counts=True)

        # create a potential values for unique elements:
        for i in range(0, len(uniqTags), 1):
            for j in range(0, len(self.x), 1):
                if self.tag[j] == uniqTags[i]:
                    self.ipot[j] = i + 1
        self.sortByDistance()
        self.ipot[0] = 0
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')
        header0 = ' POTENTIALS\n *    ipot   Z  element            l_scmt  l_fms   stoichiometry\n'
        file.write(header0)
        line0 = '\t\t{0}\t{1}\t\t{2}\t\t\t\t\t{3}\t\t{4}\t\t{5:.5f}\n' \
            .format(0, pd.elements.symbol(self.tag[0]).number, self.tag[0],
                    self.getLscmtLfms(self.tag[0])[0], self.getLscmtLfms(self.tag[0])[1], 0.001)  # check for 0 pot
        # line0 = '\t\t{0}\t{1}\t\t{2}\t\t\t\t\t{3}\t\t{4}\t\t{5:.5f}\n'\
        #     .format(0, pd.elements.symbol(self.tag[0]).number, self.tag[0],
        #             self.getLscmtLfms(0)[0],self.getLscmtLfms(0)[1], 1/(len(self.x))) #check for 0 pot
        file.write(line0)
        for i in range(0, len(uniqTags), 1):
            if uniqTags[i] != self.tag[0]:
                line = '\t\t{0}\t{1}\t\t{2}\t\t\t\t\t{3}\t\t{4}\t\t{5:.5f}\n' \
                    .format(i + 1, pd.elements.symbol(uniqTags[i]).number, uniqTags[i],
                            self.getLscmtLfms(uniqTags[i])[0], self.getLscmtLfms(uniqTags[i])[1],
                            self.getStoichiometry(uniqTags[i]))
                file.write(line)
            else:
                # if uniq element is equal to the Central Atom check the number of that atoms.
                # If it will be more then 1 we will write new line to the potential table:
                if uniqTagsCounts[i] > 1:
                    line = '\t\t{0}\t{1}\t\t{2}\t\t\t\t\t{3}\t\t{4}\t\t{5:.5f}\n' \
                        .format(i + 1, pd.elements.symbol(uniqTags[i]).number, uniqTags[i],
                                self.getLscmtLfms(uniqTags[i])[0], self.getLscmtLfms(uniqTags[i])[1],
                                self.getStoichiometry(uniqTags[i]))
                    file.write(line)

    def writeFeffInpFileSeq(self, ext='inp'):
        '''
        # write feff input header lines, table of potentials and table
        # coordinates and attributes of atoms to file
        :param filename: full path to output file
        :param mask: base name of output file (ex:"test_000001.txt" )
        :return:
        '''
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        mask = self.outNameFEFF + '_'
        self.backUpStruct()
        folder = self.outDirFEFF
        filename = os.path.join(folder, 'test.' + ext)
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            writeHeaderToInpFile(filename=fn)
            self.ipotTableConstructor(filename=fn, overwriteOrNot=False)
            self.writeTableToFile(filename=fn, overwriteOrNot=False)
            # print('')
            self.restoreFromBackUp()

    def backUpStruct(self):
        # save current variables to *tmp variables
        self.tmp_x = self.x
        self.tmp_y = self.y
        self.tmp_z = self.z
        self.tmp_ipot = self.ipot
        self.tmp_tag = self.tag
        self.tmp_r = self.r
        self.tmp_atomIndex = self.atomIndex

    def restoreFromBackUp(self):
        # restore saved variables form tmp to current
        self.x = self.tmp_x
        self.y = self.tmp_y
        self.z = self.tmp_z
        self.ipot = self.tmp_ipot
        self.tag = self.tmp_tag
        self.r = self.tmp_r
        self.atomIndex = self.tmp_atomIndex

    def writeXYZfileSeq(self, ext='xyz'):
        '''
        write .xyz structure file header(number of atoms in structure)
        and atoms tag with XYZ positions
        :param outdir: local project folder
        :param mask: basename of output file (ex:'test_000006.xyz')
        :param ext: output file extension
        :return:
        '''
        mask = self.outNameXYZ + '_'
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        folder = self.outDirXYZ
        filename = os.path.join(folder, 'test.' + ext)
        self.backUpStruct()
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            self.writeTableToXYZfile(filename=fn, overwriteOrNot=True)
            self.restoreFromBackUp()

    def getLscmtLfms(self, askTag='Mn'):
        '''
        get l_scmt and l_fms for potential tab (in feff.inp) from database
        :param index: index of element which we are looking for
        :return: array with two values: l_scmt and l_fms
        '''
        tag = np.array(['Ga', 'As', 'Mn', ])
        lscmt = [4, 4, 3, ]
        lfms = [4, 4, 3, ]
        tagIndexLscmt = dict(zip(tag, lscmt))
        tagIndexLfms = dict(zip(tag, lfms))
        return tagIndexLscmt[askTag], tagIndexLfms[askTag]

    def getStoichiometry(self, tag='Mn'):
        '''
        get stoichiometry for potential tab (in feff.inp) from database
        :param index: index of element which we are looking for
        :return:  stoichiometry
        '''
        unique, counts = np.unique(self.tag, return_counts=True)
        elCount = dict(zip(unique, 100 * counts / len(self.x)))
        return elCount[tag]

    def createUniqDirOut(self, projectDir='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test'):
        """
        create output directories for feff.inp, .xyz and RDF files
        :param projectDir: full path to local project directory
        :return:
        """
        self.outDirFEFF = create_out_data_folder(projectDir, first_part_of_folder_name='inp_')
        self.outDirXYZ = create_out_data_folder(projectDir, first_part_of_folder_name='xyz_')
        self.outDirRDF = create_out_data_folder(projectDir, first_part_of_folder_name='rdf_')
        self.outDirXSF = create_out_data_folder(projectDir, first_part_of_folder_name='xsf_')
        self.outDirCFG = create_out_data_folder(projectDir, first_part_of_folder_name='stem_')
        self.outDirAverFeffInp = create_out_data_folder(projectDir, first_part_of_folder_name='aver_inp')

        # for FEFF calculation:
        self.outDirFEFFCalc = create_out_data_folder(projectDir, first_part_of_folder_name='feff_')
        self.outDirFEFFtmp = create_out_data_folder(projectDir, first_part_of_folder_name='tmp_')

        # for PWScf (Quantum Espresso):
        self.outDirSCF = create_out_data_folder(projectDir, first_part_of_folder_name='scf_')

    def calcRDF(self):
        """
        radial distribution file calculation
        :return: array number of atoms in range into distance units
        """
        rdfArrayOut = np.zeros(len(self.rdfDist), dtype='int')
        x = self.rdfDist
        r = self.r
        for i in range(0, len(x)):
            if i < (len(x) - 1):
                # rdfArrayOut[i] = np.where( np.logical_and( r >= x[i], r < x[i+1]) )
                rdfArrayOut[i] = len(np.logical_and(r > x[i], r <= x[i + 1]).nonzero()[0])
            else:
                rdfArrayOut[i] = len((r >= x[i]).nonzero()[0])
        return rdfArrayOut

    def writeRDFtoFile(self,
                       filename='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test\\rdf.dat',
                       overwriteOrNot=True):
        # writing table of distances and number of atoms in unit distance to file
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')

        header0 = '# N atoms in structure = {0}\n'.format(len(self.x))
        header1 = '# distance\tN of atoms\n'
        file.write(header0)
        file.write(header1)
        y = self.calcRDF()
        for i in range(0, len(self.rdfDist)):
            line = '{0:.5f}\t\t{1}\n'.format(self.rdfDist[i], y[i])
            file.write(line)
        file.close()

    def writeRDFfileSeq(self, ext='dat'):

        mask = self.outNameRDF + '_'
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        folder = self.outDirRDF
        filename = os.path.join(folder, 'test.' + ext)
        self.backUpStruct()
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            self.writeRDFtoFile(filename=fn, overwriteOrNot=True)
            self.restoreFromBackUp()

    def writeStructToXSF(self,
                         filename='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test\\test.xsf',
                         overwriteOrNot=True):
        # writing input structure file for XCrySDen(*.xsf)
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')

        header0 = 'CRYSTAL\nPRIMVEC\n'
        file.write(header0)
        for line in self.primvec:
            file.write(line)
        header1 = 'PRIMCOORD\n{0} 1\n'.format(len(self.x))
        file.write(header1)
        for i in range(0, len(self.x), 1):
            line = '\t{0}\t{1:+.5f}\t{2:+.5f}\t{3:+.5f}\n' \
                .format(pd.elements.symbol(self.tag[i]).number, self.x[i], self.y[i], self.z[i]).replace('+', ' ')
            file.write(line)
        file.close()

    def writeXSFfileSeq(self, ext='xsf'):

        mask = self.outNameXSF + '_'
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        folder = self.outDirXSF
        filename = os.path.join(folder, 'test.' + ext)
        self.backUpStruct()
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            self.writeStructToXSF(filename=fn, overwriteOrNot=True)
            self.restoreFromBackUp()

    def writeStructToCFG(self,
                         filename='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test\\test.cfg',
                         overwriteOrNot=True):
        # writing input structure file for QSTEM(*.cfg)
        if overwriteOrNot:
            file = open(filename, 'w')
        else:
            file = open(filename, 'a+')
        header = []
        header.append('Number of particles = {0}\n'.format(len(self.x)))
        header.append('A = 1.0 Angstrom (basic length-scale)\n')
        header.append(
            'H0(1,1) = {0} A\nH0(1,2) = {1} A\nH0(1,3) = {2} A\n'.format(self.HO[0, 0], self.HO[0, 1], self.HO[0, 2], ))
        header.append(
            'H0(2,1) = {0} A\nH0(2,2) = {1} A\nH0(2,3) = {2} A\n'.format(self.HO[1, 0], self.HO[1, 1], self.HO[1, 2], ))
        header.append(
            'H0(3,1) = {0} A\nH0(3,2) = {1} A\nH0(3,3) = {2} A\n'.format(self.HO[2, 0], self.HO[2, 1], self.HO[2, 2], ))
        header.append('.NO_VELOCITY.\nentry_count = 3\n')
        for line in header:
            file.write(line)

        uniqTags = np.unique(self.tag)
        for tag in uniqTags:
            file.write('{0}\n{1}\n'.format(round(pd.elements.symbol(tag)._mass), tag))
            for i in range(len(self.x)):
                if self.tag[i] == tag:
                    line = '{0:+.5f}\t{1:+.5f}\t{2:+.5f}\n' \
                        .format(self.x[i] / self.HO[0, 0], self.y[i] / self.HO[1, 1],
                                self.z[i] / self.HO[2, 2]).replace('+', ' ')
                    file.write(line)
        file.close()

    def writeCFGfileSeq(self, ext='cfg'):

        mask = self.outNameCFG + '_'
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        folder = self.outDirCFG
        filename = os.path.join(folder, 'test.' + ext)
        self.backUpStruct()
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            self.writeStructToCFG(filename=fn, overwriteOrNot=True)
            self.restoreFromBackUp()

    def calcAndPlotMeanRDF(self):
        self.dataPath = self.outDirRDF
        self.loadFiles()
        self.plot()

    def writeAverFeffInpFile(self, ext='inp'):
        '''
        # write average feff input header lines, table of potentials and table
        # coordinates and attributes of atoms to file
        :param filename: full path to output file
        :param mask: base name of output file (ex:"test_000001.txt" )
        :return:
        '''
        numOfRepeat = self.tag.tolist().count(self.majorElemTag)
        mask = self.outNameAverFeffInp + '_'
        self.backUpStruct()
        folder = self.outDirAverFeffInp
        filename = os.path.join(folder, 'test.' + ext)
        vectorIndex = (self.tag == self.majorElemTag).nonzero()[0].tolist()
        for i in range(0, numOfRepeat, 1):
            self.centrShift(centrAtomIndex=vectorIndex[i])
            fn = createUniqFile(filename, mask=mask)
            writeHeaderToInpFile(filename=fn)
            self.ipotTableConstructor(filename=fn, overwriteOrNot=False)
            self.writeTableToFile(filename=fn, overwriteOrNot=False)
            # print('')
            self.restoreFromBackUp()


if __name__ == "__main__":
    a = Unitcell(6)
    a.x[:] = [0, 1, 2, 3, 4, 5]
    a.y[:] = [1, 2, 3, 4, 5, 6]
    a.z[:] = [1, 0, 1, 0, 1, 0]
    a.tag[:] = ['Ga', 'As', 'Mn', 'As', 'Mn', 'Ga', ]
    a.writeStructToCFG()
    # a.centrShift()
    #
    # a.createUniqDirOut()
    # a.writeFeffInpFileSeq()
    # a.writeXYZfileSeq()
    # a.writeRDFfileSeq()
