import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libs.dir_and_file_operations import listOfFilesFN_with_selected_ext, deleteAllSelectExtenFilesInFolder
from scipy import integrate


class AverageRDF ():
    def __init__(self):
        self.structName = 'RDF'
        self.dataPath ='C:\\wien2k_ver14\\VirtualBoxShare\\MD_FEFF_sim\\test\\rdf__0001'
        self.R =[]
        self.matrixOfRDF =[]
        self.meanRDF = []
        self.stdRDF = []

    def loadFiles(self):
        filesFullPathName = listOfFilesFN_with_selected_ext(self.dataPath, ext = 'dat')
        numOfColumns = len(filesFullPathName)
        data = np.loadtxt(filesFullPathName[0], float)
        self.R = data[:,0]
        self.meanRDF = np.zeros(len(self.R))
        self.stdRDF = np.zeros(len(self.R))
        self.matrixOfRDF = np.zeros((len(self.R), numOfColumns))
        self.matrixOfRDF[:,0] = data[:,1]
        for i in range(1,numOfColumns,1):
            data = np.loadtxt(filesFullPathName[i])
            self.matrixOfRDF[:,i] = data[:,1]
        self.meanRDF = np.mean(self.matrixOfRDF,axis=1)
        self.stdRDF = np.std(self.matrixOfRDF,axis=1)
        outArray = np.zeros((len(self.R),3))
        outArray[:,0] = self.R
        outArray[:,1] = self.meanRDF
        outArray[:,2] = self.stdRDF
        h = 'distance\tmean N of Atoms\tstd'
        np.savetxt(os.path.join(self.dataPath, '0_mean.txt'),
                          outArray, header=h)

    def plot(self):
        plt.figure(figsize=(10,8), dpi = 96)
        plt.plot(self.R, self.meanRDF, ls = '-', c = 'k', lw = 1.5)
        plt.fill_between(self.R, self.meanRDF-self.stdRDF, self.meanRDF+self.stdRDF,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=4, linestyle='dashdot', antialiased=True)
        plt.ylabel('Number of atoms', fontsize=18)
        plt.axis([2, 7, 0, np.max(self.meanRDF[self.R<7]+1)])
        plt.twinx()
        plt.plot(self.R, integrate.cumtrapz(self.meanRDF, initial = 0), c = 'r')
        plt.axis([2, 7, 0, 40])
        plt.xlabel('Distance ($\AA$)', fontsize=18)
        plt.ylabel('Integrated number of atoms', fontsize=16)
        plt.grid(True)
        plt.title('$RDF$ for structure: ' + self.structName + '\n after $GULP$ $MD$ relaxation procedure', fontsize=22)
        plt.gcf().savefig(os.path.join(self.dataPath, '0_mean.png'))
        # plt.show()
        # print('_')
        deleteAllSelectExtenFilesInFolder(self.dataPath, ext = 'dat')
        print('All rdf *.dat files were deleted')


if __name__ == "__main__":
    b = AverageRDF()
    b.dataPath = 'C:\\wien2k_ver14\\VirtualBoxShare\\results_Pasha\\test_less SnpSht\\rdf__0001'
    b.loadFiles()
    b.plot()