import matplotlib.pyplot as plt
import tomopy
import numpy as np
import radonusfft

# numpy array are sent to C code by pointers
def getp(a):
    return a.__array_interface__['data'][0]

# test 
N = 512 # object size in 1 dimension
Ntheta = 256 # number of angles
Nz = 16 # number of slices
center = N/2+3 # rotation center
theta = np.float32(np.arange(0, Ntheta)*np.pi/Ntheta) # angles

# create class for the transform with allocating memory on GPU
cl = radonusfft.radonusfft(getp(theta), center, Ntheta, Nz, N)

# init data as a complex array
f0 = tomopy.misc.phantom.shepp2d(N)+1j*np.fliplr(tomopy.misc.phantom.shepp2d(N))
f = np.tile(f0,[Nz,1,1])
# projection
g = np.zeros([Ntheta, Nz, N], dtype="complex64")# memory for result
cl.fwd(getp(g), getp(f))
# inversion
ff = np.zeros([Nz, N, N], dtype="complex64")# memory for result
cl.adj(getp(ff), getp(g))

# circ mask
ff = tomopy.misc.corr.circ_mask(ff.real,axis=0)+1j*tomopy.misc.corr.circ_mask(ff.imag,axis=0)
# plot
plt.subplot(2, 4, 1)
plt.title("init real part")
plt.imshow(np.squeeze(f[0, :, :].real))
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title("init imag part")
plt.imshow(np.squeeze(f[0, :, :].imag))
plt.colorbar()
plt.subplot(2, 4, 2)
plt.title("projection real part")
plt.imshow(np.squeeze(g[:, 0, :].real))
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title("projection imag part")
plt.imshow(np.squeeze(g[:, 0, :].imag))
plt.colorbar()
plt.subplot(2, 4, 3)
plt.title("reconstruction real part")
plt.imshow(np.squeeze(ff[0, :, :].real))
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title("reconstruction imag part")
plt.imshow(np.squeeze(ff[0, :, :].imag))
plt.colorbar()
plt.subplot(2, 4, 4)
plt.title("error real part")
plt.imshow(np.squeeze(ff[0, :, :].real-f[0, :, :].real))
plt.colorbar()
plt.subplot(2, 4, 8)
plt.title("error imag part")
plt.imshow(np.squeeze(ff[0, :, :].imag-f[0, :, :].imag))
plt.colorbar()

plt.show()
