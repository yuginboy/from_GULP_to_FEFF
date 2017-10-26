'''
* Created by Pavlo Konstantynov
'''
line0 = ((' HOLE 1   1.0   *  Mn K edge , MniB-1Mn-0V\n\n'),
         (' *         mphase,mpath,mfeff,mchi\n'),
         (' CONTROL   1      1     1     1\n'),
         (' PRINT     1      0     0     0\n\n'),
         (' RMAX       10.0\n\n'),
         (' *CRITERIA     curved   plane\n'),
         (' *DEBYE        temp     debye-temp\n'),
         (' NLEG         4\n\n'))
def writeHeaderToInpFile(filename = 'test.txt'):
    file = open(filename,'w')
    for i in line0:
        file.write(i)
