import numpy as np
import healpy as hp
from PixelILC import doPixelILC

#TQUmap will be a numpy array with the shape [Nfreqs,3,npix]

def cmb(nu):
	x = 0.0176086761 * nu
	ex = np.exp(x)
	sed = ex * (x / (ex - 1)) ** 2
	return sed

freqs = np.array([27,39,93,145,225,280])
Nfreqs = 6
nside = 1024
npix = 12*nside**2
fwhm = np.radians(2.0/60.0) # in radians, for the weight of the pixel domain
output_file = 'output/test.fits'

vec = np.array([0,0,1])#hp.ang2vec(np.radians(10.0),np.radians(90))
pp = hp.query_disc(nside,vec,np.radians(0.25))
mask = np.zeros(npix)
mask[pp] = 1.0
npix_mask = int(np.sum(mask))
indices = np.where(mask==1.0)

print('sky fraction %.5f, which is %i pixels'%(npix_mask/float(npix),npix_mask))
print(indices[0].shape)
#exit()

TQUmaps = np.zeros((Nfreqs,3,npix))
a = cmb(freqs)
a = a / a[0]

for n,freq in enumerate(freqs):
	TQUmaps[n,:,:] = hp.read_map('../../sigmar_forecast/Create_simulated_maps_pysm3/files/FullMap_wPysm3_ns1024_tqu_f%ip0_uK_RJ_10arcminSmoothed_SO_White.fits'%freq,field=(0,1,2),verbose=False)

w_ilc = doPixelILC(TQUmaps, TQUmaps, nside, a, fwhm, Nfreqs, indices[0], len(indices[0]), 0 )

recons_cmb = np.zeros((3,npix))
for k in range(3):
	counter = 0
	for pi in range(npix):
		if mask[pi] == 1.0:
			recons_cmb[k,pi] = np.matmul(w_ilc[counter,k,:],TQUmaps[:,k,pi])
			counter += 1

#np.savez(output_file+'.npz',w_ilc=w_ilc)
#hp.write_map(output_file,recons_cmb,overwrite=True)