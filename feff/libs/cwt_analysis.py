#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LightSource
plt.rcParams.update({'font.size': 12})

# f = open(r'C:\Users\melikhov\Desktop\GAMNAS_CWT\350.chik')
f = open(r'/home/yugin/VirtualboxShare/GaMnO/1mono1SR2VasVga2_6/feff__0001/result_1mono1SR2VasVga2_6.txt')
file = np.loadtxt(f, skiprows=96)
# file = np.loadtxt(f, skiprows=37)
kex = file[:, 0]
chiex = file[:, 1]

def cauchy_wavelet(k, chi, rmax_out=10, nfft=2048):
    """
    Cauchy Wavelet Transform for XAFS, following work of Munoz, Argoul, and Farges

    Parameters:
    -----------
      k:        1-d array of photo-electron wavenumber in Ang^-1 or group
      chi:      1-d array of chi
      group:    output Group
      rmax_out: highest R for output data (10 Ang)
      kweight:  exponent for weighting spectra by k**kweight
      nfft:     value to use for N_fft (2048).

      Returns:
    ---------
      None   -- outputs are written to supplied group.

    Notes:
    -------
    Arrays written to output group:
    r                  uniform array of R, out to rmax_out.
    wcauchy            complex cauchy wavelet(k, R)
    wcauchy_mag        magnitude of wavelet(k, R)
    wcauchy_re         real part of wavelet(k, R)
    wcauchy_im         imaginary part of wavelet(k, R)

    Supports First Argument Group convention (with group
    member names 'k' and 'chi')

    """
    kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    rstep = (np.pi/2048)/kstep
    rmin = 1.e-7
    rmax = rmax_out
    nrpts = int(np.round((rmax-rmin)/rstep))
    nkout = len(k)
    kweight = 1
    chi = chi * k**kweight

    # extend EXAFS to 1024 data points...
    NFT = int(nfft/2)
    if len(k) < NFT:
        knew = np.arange(NFT) * kstep
        xnew = np.zeros(NFT) * kstep
        xnew[:len(k)] = chi
    else:
        knew = k[:NFT]
        xnew = chi[:NFT]

    # FT parameters
    freq = (1.0/kstep)*np.arange(nfft)/(2*nfft)
    omega = 2*np.pi*freq

    # simple FT calculation
    tff = np.fft.fft(xnew, n= 2*nfft)

    # scale parameter
    r  = np.linspace(0, rmax, nrpts)
    r[0] = 1.e-19
    a  = nrpts/(2*r)

    # Characteristic values for Cauchy wavelet:
    cauchy_sum = np.log(2*np.pi) - np.log(1.0+np.arange(nrpts)).sum()

    # Main calculation:
    out = np.zeros(nkout*nrpts,
                   dtype='complex128').reshape(nrpts, nkout)
    for i in range(nrpts):
        aom = a[i]*omega
        aom[np.where(aom==0)] = 1.e-19
        filt = cauchy_sum + nrpts*np.log(aom) - aom
        tmp  = np.conj(np.exp(filt))*tff[:nfft]
        out[i, :] = np.fft.ifft(tmp, 2*nfft)[:nkout]

    rout  =  r
    wcauchy =  out
    wcauchy_mag =  np.sqrt(out.real**2 + out.imag**2)
    wcauchy_re =  out.real
    wcauchy_im =  out.imag
    return rout, wcauchy, wcauchy_mag, wcauchy_im, wcauchy_re

cwtout = cauchy_wavelet(kex, chiex)

deg = u'\xb0'
plt.imshow(np.abs(cwtout[1]),extent = [min(kex),max(kex),
                                       max(cwtout[0]),min(cwtout[0])], aspect='auto')
plt.xlabel(r'$k^{-1} \, (\AA^{-1})$')
plt.ylabel(r'$R \, (\AA) $')
plt.title('CWT analysis for $350$ '+deg+'$C$ sample')
plt.colorbar(use_gridspec=True )
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.06)

fig = plt.figure()
fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
ax = fig.gca(projection='3d')
y, x = np.meshgrid(kex, cwtout[0])
z = np.abs(cwtout[1])
# ax.plot_surface(x,y, z)

# # ls = LightSource(270, 45)
# # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=True)
ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.6)
zmax=np.max(z)
ly = max(kex) - min(kex)
lx = max(cwtout[0]) - min(cwtout[0])

cset = ax.contourf(x, y, z, zdir='z', offset=-zmax, cmap=cm.coolwarm)
# cset = ax.contourf(x, y, z, zdir='x', offset=min(cwtout[0]) - 0.1*lx, cmap=cm.coolwarm)
# cset = ax.contourf(x, y, z, zdir='y', offset=max(kex)+0.1*ly, cmap=cm.coolwarm)

# ax.view_init(elev=55, azim=210)
ax.set_xlabel(r'$R \, (\AA) $')
ax.set_xlim(min(cwtout[0]) - 0.1*lx, max(cwtout[0])+0.1*lx)
ax.set_ylabel(r'$k^{-1} \, (\AA^{-1})$')
ax.set_ylim(min(kex) - 0.1*ly, max(kex)+0.1*ly)
ax.set_zlabel('|CWT|, arb. un.')
ax.set_title('$CWT$ $analysis$ $for$ $350$ '+deg+'$C$ sample')

ax.set_zlim(-zmax, zmax)
plt.show()
