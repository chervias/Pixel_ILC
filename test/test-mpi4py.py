import numpy as np
import healpy as hp
from PixelILC import doPixelILC
from mpi4py import MPI

shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
size_pool = shared_comm.Get_size()
rank = shared_comm.Get_rank()

def cmb(nu):
	x = 0.0176086761 * nu
	ex = np.exp(x)
	sed = ex * (x / (ex - 1)) ** 2
	return sed

freqs = np.array([27,39,93,145,225,280])
Nfreqs = 6
nside = 1024
npix = 12*nside**2
fwhm = np.radians(60.0/60.0) # in radians, for the weight of the pixel domain

if rank==0:
	output_file = 'output/test-mpi.fits'
	vec = np.array([0,0,1])#hp.ang2vec(np.radians(10.0),np.radians(90))
	pp = hp.query_disc(nside,vec,np.radians(10.0))
	mask = np.zeros(npix,dtype=np.double)
	mask[pp] = 1.0
	npix_mask = int(np.sum(mask))
	print('sky fraction %.5f, which is %i pixels'%(npix_mask/float(npix),npix_mask))
else:
	mask = np.empty(npix, dtype=np.double)
# Broadcast the mask to all ranks
shared_comm.Bcast(mask, root=0)

# Now we create the TQUmaps array, but only rank==0 will fill it
double_size = MPI.DOUBLE.Get_size()
size_TQUmaps = (Nfreqs,3,npix)
if rank==0:
	total_size_TQUmaps = np.prod(size_TQUmaps)
	nbytes_TQUmaps = total_size_TQUmaps*double_size
else:
	nbytes_TQUmaps = 0

shared_comm.Barrier()

win_TQUmaps = MPI.Win.Allocate_shared(nbytes_TQUmaps, double_size, comm=shared_comm)
buf_TQUmaps, itemisize_TQUmaps = win_TQUmaps.Shared_query(0)
TQUmaps_arr = np.ndarray(buffer=buf_TQUmaps, dtype=np.double, shape=size_TQUmaps)
win_TQUmaps.Fence()
# rank 0 will only fill the array
if rank==0:
	#TQUmap will be a numpy array with the shape [Nfreqs,3,npix]
	TQUmaps = np.zeros((Nfreqs,3,npix))
	for n,freq in enumerate(freqs):
		TQUmaps[n,:,:] = hp.read_map('../../sigmar_forecast/Create_simulated_maps_pysm3/files/FullMap_wPysm3_ns1024_tqu_f%ip0_uK_RJ_10arcminSmoothed_SO_White.fits'%freq,field=(0,1,2),verbose=False)
	win_TQUmaps.Put(TQUmaps,0,0)
win_TQUmaps.Fence()
shared_comm.Barrier()

# This is the CMB SED, every rank will have a copy
a = cmb(freqs)
a = a / a[0]

# Now we need to determine which pixels are inside the mask and split the array
# This is what I used to do this https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array/36082684#36082684
if rank == 0:
	indices = np.where(mask==1.0)
	test = indices[0].astype('int32')
	split = np.array_split(test, size_pool)
	split_size = [len(split[i]) for i in range(len(split))]
	split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]
	# This is the array to receive the total weights from all the ranks
	w_ilc_tot = np.zeros((npix_mask,3,Nfreqs),dtype='d')
	split_wilc = np.array_split(w_ilc_tot,size_pool, axis = 0) #Split input array by the number of available cores
	split_sizes_wilc = []
	for i in range(0,len(split_wilc),1):
		split_sizes_wilc = np.append(split_sizes_wilc, len(split_wilc[i]))
	split_sizes_wilc_output = split_sizes_wilc*3*Nfreqs
	displacements_wilc_output = np.insert(np.cumsum(split_sizes_wilc_output),0,0)[0:-1]
else:
	test = None
	split = None
	split_size = None
	split_disp = None
	w_ilc_tot = None
	split_wilc = None
	split_sizes_wilc = None
	split_sizes_wilc_output = None
	displacements_wilc_output = None

split_size = shared_comm.bcast(split_size, root = 0)
split_disp = shared_comm.bcast(split_disp, root = 0)
test_local = np.zeros(split_size[rank],dtype='int32')
shared_comm.Scatterv([test, split_size, split_disp, MPI.INT], test_local, root=0)
# Now in test_local we have the corresponding pixels that each rank will process
#print('Rank %i will process '%rank,test_local.shape)

split_sizes_wilc_output = shared_comm.bcast(split_sizes_wilc_output, root = 0)
displacements_wilc_output = shared_comm.bcast(displacements_wilc_output, root = 0)

#print('The first element of rank %i is %E'%(rank,TQUmaps_arr[0,1,12873]))
w_ilc_rank = doPixelILC(TQUmaps_arr, nside, a, fwhm, Nfreqs, test_local, len(test_local) , rank)

shared_comm.Barrier()
# Now we need to Gatherv the w_ilc arrays, w_ilc is the send buffer
shared_comm.Gatherv(w_ilc_rank,[w_ilc_tot,split_sizes_wilc_output,displacements_wilc_output,MPI.DOUBLE], root=0)

if rank==0:
	print("Final shape of weights",w_ilc_tot.shape)
	recons_cmb = np.zeros((3,npix))
	for k in range(3):
		counter = 0
		for pi in range(npix):
			if mask[pi] == 1.0:
				recons_cmb[k,pi] = np.matmul(w_ilc_tot[counter,k,:],TQUmaps_arr[:,k,pi])
				counter += 1
	np.savez(output_file+'.npz',w_ilc=w_ilc_tot)
	hp.write_map(output_file,recons_cmb,overwrite=True)