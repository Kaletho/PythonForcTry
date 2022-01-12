import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import scipy.linalg
import itertools
from scipy import interpolate
import pylab as py
import matplotlib.colors as colors

# "Import" the file properly
fileName='a6Cali-Big'
# The next is the first tried file. Don't change!
#f = pd.read_csv('niNWsInMembraneFORC5kOe_3', header=None, skiprows=85, skipfooter=1, skip_blank_lines=False, engine='python', keep_default_na=False, na_values=[''])
f = pd.read_csv(fileName, header=None, skiprows=85, skipfooter=2, skip_blank_lines=False, engine='python', keep_default_na=False, na_values=[''])
#f = pd.read_csv('test', header=None, skiprows=85, skipfooter=1, skip_blank_lines=False, engine='python', keep_default_na=False, na_values=[''])
# Use the next two lines to plot only
g = pd.read_csv(fileName, names=['Field(Oe)','M(emu)'], skiprows=86, skipfooter=2, skip_blank_lines=False, engine='python', keep_default_na=False, na_values=[''])
g.set_index('Field(Oe)', inplace=True)

# Check if everything went well
#print(f.head())   # Prints first 5 rows
h = plt.figure()
plt.plot(g)
plt.grid(which='both', linewidth=2)
h.suptitle('RawData '+fileName)
plt.xlabel('Happl (Oe)')
plt.ylabel('M (emu)')
h.savefig('rawData.png', dpi=300)
#plt.show()

# Initialize a list and convert to array later
#forc = []
Ha = []
Hb = []

# Take from measurement parameters
Hcal = +6.72E+03 # calibration field. Take the one from the data, not the header
HNcr = +97.70125E+00 # Field increment step
Hbmin = -6.073084E+03
Hbmax = +6.7E+03

# create an artificial list for Hb
Hbp = np.arange(Hbmin, Hbmax, HNcr)

nForc = 0
x = 0
# From the datafile save Hb and Ha
for row in f.itertuples():  
#	print row[0], row[1] # iterates over all rows on each column
	# This itertuples returns strings!
	if np.isnan(row[1]) == False:
		if float(row[0]) < Hcal: # Ignore calibration field lines
		#if abs(float(row[0])) < Hcal: # Ignore calibration field lines
			Hi = float(row[0])
			if x == 0:  # Take only the first row of every FORC  
				Hr = Hi
				Ha = Ha + [Hr] # Save list of reversal fields
				Hb = Hb + [Hr] # Save first 
				x += 1
			else:
				Hb = Hb + [Hi]
		else:
			nForc += 1
	else:
		x = 0

Ha.sort()
Hb.sort()

# Hb is a mess (contains duplicates). Try to delete them:
Hb2 = [] # New Hb list
dife = [] # Check differences distribution
tol = 20.0 # Field increment tolerance
for i in range(len(Hb)-1):
	delta = abs(Hb[i]-Hb[i+1])
	if abs(delta - HNcr) < tol: # compare with the step size of the measurement
		Hb2 = Hb2 + [Hb[i]]
	dife = dife + [delta]
Hb2 = Hb2 + [Hb[len(Hb)-1]]
Hb2.sort()
#plt.plot(dife)

#print len(Ha), nForc-1, len(Hb2)
#print 'Ha: ', Ha
#print 'Hb: ', Hb2
#plt.plot(Ha)
#plt.plot(Hb2)
#plt.show()

# Now I have to correctly write M as [[M(Hb0), M(Hb1),...,M(Hb131)](Ha0), [M(Hb0), M(Hb1),...,M(Hb131)](Ha1),...,[M(Hb0), M(Hb1),...,M(Hb131)](Ha99)]
# I'll have to iterate again through the file and compare with the Hb2 list to assign the correct 'positions' for M
m = [] # m values of each FORC (will reset)
M = [] # Total magnetization vector (this will be huge!)
l = 0
for row in f.itertuples():
	if np.isnan(row[1]) == False:
		if float(row[0]) < Hcal:
		#if abs(float(row[0])) < Hcal:
			if m == []:
#				print float(row[0])
				for j in range(len(Hb2)): 
					del2 = abs(float(row[0]) - Hb2[j])
					if del2 < 5.0:
#						print del2, float(row[0]), Hb2[j], float(row[1]), len(m)
						l += 1
						break
					else:
						m = m + [0.0]  # Artificial points to complete dimension 1 of M array lengths have to match (check below print (way below))
						l += 1
			m = m + [float(row[1])]
			l += 1
			lastm = m[:] # this is higly inefficient
			#print m
		elif l != 0:
			for j in range(len(Hb2)-l+1):
				m = m + [0.0]
			l = 0
			M.append(m)
			m = []
# need to append the trailing zeroes to lastm 
for z in range(len(Hb2)-len(lastm)):
	lastm = lastm + [0.0]
M.append(lastm) # if its stupid but it works...

M.reverse() # Because Ha and Hb are sorted from min to max

#print 'M[0]: ', M[0], 'Len M: ', len(M)
#print 'M[last]', M[len(Ha)-1] 
#plt.plot(Hb2, M[0])
#print len(M)

print 'Check: all dimmensions should match'
print 'len(Ha) = len(M) = nForc-1; len(M[i]) = len(Hb)'
print 'Ha: ', len(Ha)
print 'nFORCS: ', nForc-1
print 'M: ', len(M)
print 'Hb: ', len(Hb2)
print 'm: ', len(M[0])

# For the contour plots
cmin = 0.0
cmax = 2.5e-6
cstep = 1.0e-8
cdpi = 300

# This plots the magnetization correclty 
def plotM(pHb, pHa, pM, title):
	HB, HA = np.meshgrid(pHb,pHa)

	g=plt.figure()
	bounds = np.arange(cmin,cmax,cstep)
	norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
	plt.contourf(HB, HA, pM, cmap=plt.cm.jet, extend='both', levels=bounds, norm=norm)
	g.suptitle(title+' '+fileName)
	plt.xlabel('Hb (Oe)')
	plt.ylabel('Ha (Oe)')
	plt.colorbar(format='%.0e')
	g.savefig(title+'.png')

# This plots the rotated coordinates correclty
def plotMR(Hc, Hu, Mv, title):
	g=plt.figure()
	bounds = np.arange(cmin,cmax,cstep)
	norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
	#plt.tricontourf(Hc, Hu, Mv, cmap=plt.cm.jet, extend='both', levels=np.arange(-5e-8,5e-8,0.5e-8))
	plt.tricontourf(Hc, Hu, Mv, cmap=plt.cm.jet, extend='both', levels=bounds, norm=norm)
	g.suptitle(title+' '+fileName)
	plt.xlabel('Hc (Oe)')
	plt.ylabel('Hu (Oe)')
	plt.colorbar(format='%.0e')
	g.savefig(title+'.png', dpi=cdpi)

def subGrid(Hb, Ha, M, SF, i, j):
	'''
	It works!
	This will be highly ineficcient, but hopefully will work
	This subgrid will be used by fitSurf to find each coefficient
	Returns a (2*SF + 1)x(2*SF + 1) grid of Hb, Ha, M as arrays
	The grid is around point i,j
	'''
	x = []
	y = []
	Mg = []
	dum = [] # dummy
	check = [] # array to check if M[j][i] == 0.0 
	xstrt = i - SF
	xstp = i + SF + 1
	ystrt = j - SF
	ystp = j + SF + 1

	xdel = xstp - xstrt
	ydel = ystp - ystrt

	# Need some if to account for the edges
	if xstrt < 0:
		xstrt += SF 
		xstp += SF
	if ystrt < 0:
		ystrt += SF 
		ystp += SF
	if xstp > len(Hb):
		xstrt -= SF 
		xstp -= SF
	if ystp > len(Ha):
		ystrt -= SF   
		ystp -= SF  

	for l in range(xstrt, xstp, 1):	
		x.append(Hb[l]) 
	for l in range(ystrt, ystp, 1):	
		y.append(Ha[l])
	for l in range(ystrt, ystp, 1):	
		for m in range(xstrt, xstp, 1): # x is in the second dimension
			dum.append(M[l][m])
		Mg.append(dum)
		dum = []
	
	return x, y, Mg

# This should be a subgrid around 'points' 65,35
#sHb, sHa, sM = subGrid(Hb2, Ha, M, 3, 65, 45)
#plotM(Hb2, Ha, M, 'Magnetization (emu)')
#plotM(sHb, sHa, sM, 'Magnetization (emu) - SubGrid')
#print sM
#plt.show()
#raw_input()

def surfFit(sHb, sHa, sM):
	'''
	It works!
	Got this from here:
	https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
	'''
	X,Y = np.meshgrid(sHb, sHa)

	# Need to write a matrix in the form A = [[x1,y1,z1], [x2,y2,z2],...]
	dum = []
	for i in range(len(sHb)):
		for j in range(len(sHa)):
#			if sM[j][i] != 0.0: # Ignore artificial points 
			dum.append([sHb[i], sHa[j], sM[j][i]]) # in M, x is the second dimension

	Mp = np.array(dum)
	# The following code is not mine
	# --------- START COPIED CODE ------------
	XX = X.flatten()
	YY = Y.flatten()

	# best-fit quadratic curve
	A = np.c_[np.ones(Mp.shape[0]), Mp[:,:2], np.prod(Mp[:,:2], axis=1), Mp[:,:2]**2] # original
	C,_,_,_ = scipy.linalg.lstsq(A, Mp[:,2])
	# This is in the form
	# C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]

	# evaluate it on a grid
	# if you want to check if fitting went well
	Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
	
	return C, Z


def chanCoord(Hb, Ha, M):
	'''
	Return Hc, Hu, and M to be plotted
	M can be any 2D array
	'''
	dum = []
	for j in range(len(Ha)):
		for i in range(len(Hb)):
			if M[j][i] != 0.0: # Ignore artificial points
				#dum.append([0.5*(Hb[i]-Ha[j]), 0.5*(Ha[j]+Hb[i]), M[j][i]]) 
				dum.append([0.5*(Hb[i]-Ha[j]), -0.5*(Ha[j]+Hb[i]), M[j][i]]) # Vazquez
			

	# It seems numpy does this easily
	A = np.asarray(dum)
	Hc = A[:,0]
	Hu = A[:,1]
	Mv = A[:,2]

	return Hc, Hu, Mv

#Hc, Hu, Mv = chanCoord(Hb2, Ha, M)
#print 'Hc, Hu, Mv'
#print len(Hc), len(Hu), len(Mv)
#plotMR(Hc, Hu, Mv, 'New Coords')

# Now I have to iterate on all the points and save -C[3] as an array like M
def forc(Hb, Ha, M, SF):
	p = [] # read as rho, -d2M/dHa dHb
	px = []
	mfit = []
	mmx = []
	for j in range(len(Ha)):
		for i in range(len(Hb)):
			if M[j][i] != 0.0: # Ignore artificial points.
				sHb, sHa, sM = subGrid(Hb, Ha, M, SF, i, j)
				C, mm = surfFit(sHb, sHa, sM)
				#px.append(-C[3])
				px.append(-0.5*C[3]) # Vazquez
				mmx.append(mm[len(sHa)/2][len(sHb)/2])
			else:
				px.append(0.0)
				mmx.append(0.0)
		p.append(px)
		px = []
		mfit.append(mmx)
		mmx = []

#	pabs = np.fabs(p) # abs() every value
#	maxp = np.amax(p)
#	p = pabs/maxp
	return p, mfit

#sHbt, sHat, sMt = subGrid(Hb, Ha, M, 2, 50, 10)
#_,Z = surfFit(sHbt, sHat, sMt)
#plotM(sHbt, sHat, sMt, 'RAW')
#plotM(sHbt, sHat, Z, 'FITTED!')

p1, mm1 = forc(Hb2, Ha, M, 1)
pHc1, pHu1, pv1 = chanCoord(Hb2, Ha, p1)
plotMR(pHc1, pHu1, pv1, 'FORC SF=1')
#plotM(Hb2, Ha, p1, 'FORC SF=1')
#plotM(Hb2, Ha, mm1, 'MFIT SF=1')

p2, mm2 = forc(Hb2, Ha, M, 2)
pHc2, pHu2, pv2 = chanCoord(Hb2, Ha, p2)
plotMR(pHc2, pHu2, pv2, 'FORC SF=2')
#plotM(Hb2, Ha, p2, 'FORC SF=2')
#plotM(Hb2, Ha, mm2, 'MFIT SF=2')

p3, mm3 = forc(Hb2, Ha, M, 3)
pHc3, pHu3, pv3 = chanCoord(Hb2, Ha, p3)
plotMR(pHc3, pHu3, pv3, 'FORC SF=3')
#plotM(Hb2, Ha, p3, 'FORC SF=3')
#plotM(Hb2, Ha, mm3, 'MFIT SF=3')

p4, mm4 = forc(Hb2, Ha, M, 4)
pHc4, pHu4, pv4 = chanCoord(Hb2, Ha, p4)
plotMR(pHc4, pHu4, pv4, 'FORC SF=4')
#plotM(Hb2, Ha, p3, 'FORC SF=3')
#plotM(Hb2, Ha, mm3, 'MFIT SF=3')

print 'DONE!'
#plt.show()
