import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(T, It, rect, p0 = np.zeros(2)):
	# Input: 
	#	T: 		Template image
	#	It: 	Current image
	#	rect: 	Current position of the car (top, left, bottom, right -> coordinates)
	#	p0: 	Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: 		Movement vector [dp_x, dp_y]
	
	# Note: Reason for bivariate spline approximation:
		# 1. Image is in discrete space.
		# 2. Location of template is in continuous space.
		# 3. To handle continuous space and tracking, we find approximate image.
	
	threshold = 0.1

	size_x = abs(rect[3] - rect[1])+1
	size_y = abs(rect[2] - rect[0])+1

	x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
	
	x = np.arange(0, It.shape[0], 1)
	y = np.arange(0, It.shape[1], 1)

	# Find gradients of image.
	Iy, Ix = np.gradient(It)
	spline_gx = RectBivariateSpline(x, y, Ix)	# Fit dI/dx in a bivariate spline.
	spline_gy = RectBivariateSpline(x, y, Iy)	# Fit dI/dy in a bivariate spline.

	spline_T = RectBivariateSpline(x, y, T)		# Fit template in a bivariate spline.
	spline_It = RectBivariateSpline(x, y, It)	# Fit current frame in a bivariate spline.

	# Find template as per the new coordinates (rect).
	c = np.linspace(x1, x2, size_x)
	r = np.linspace(y1, y2, size_y)
	cc, rr = np.meshgrid(c, r)
	T = spline_T.ev(rr, cc)

	dp = 1
	while np.square(dp).sum() > threshold:
		# Find new coordinates of image
		px, py = p0[0], p0[1]
		x1_w, y1_w, x2_w, y2_w = x1+px, y1+py, x2+px, y2+py	
	
		# Warp the Image.
		cw = np.linspace(x1_w, x2_w, size_x)
		rw = np.linspace(y1_w, y2_w, size_y)
		ccw, rrw = np.meshgrid(cw, rw)
		warpImg = spline_It.ev(rrw, ccw)
		
		# Compute error image.
		D = warpImg - T
		
		# Compute gradient
		Ix_w = spline_gx.ev(rrw, ccw)
		Iy_w = spline_gy.ev(rrw, ccw)

		# Solve least squares problem.
		A = np.array([[np.sum(Ix_w*Ix_w), np.sum(Ix_w*Iy_w)], [np.sum(Ix_w*Iy_w), np.sum(Iy_w*Iy_w)]])
		b = np.array([[-np.sum(Ix_w*D), -np.sum(Iy_w*D)]]).T
		dp = np.matmul(np.linalg.inv(A), b)
		
		#update parameters
		p0[0] += dp[0]
		p0[1] += dp[1]
		
	p = p0
	return p
