import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
	data = np.load('data/carseq.npy')
	img = data[:,:,0]
	
	# Bivariate Spline Approximation on Image.
	x = np.arange(0, img.shape[0], 1)
	y = np.arange(0, img.shape[1], 1)
	spline = RectBivariateSpline(x, y, img)
	yy, xx = np.meshgrid(y, x)

	# Generate New Image by evaluating the spline.
	c = np.arange(0, img.shape[1], 0.1)
	r = np.arange(0, img.shape[0], 0.1)
	cc, rr = np.meshgrid(c, r)
	img_new = spline.ev(rr, cc)

	plt.figure()
	plt.imshow(img_new, cmap='gray')
	plt.title('Interpolated Image, Shape: {} x {}'.format(img_new.shape[0], img_new.shape[1]))
	plt.savefig('interpolated_image.jpg')

	plt.figure()
	plt.imshow(img,cmap='gray')
	plt.title('Original Image, Shape: {} x {}'.format(img.shape[0], img.shape[1]))
	plt.savefig('original_image.jpg')

	# Make 3D plots for visualization.
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(rr, cc, img_new, color='r')
	plt.title('Plot for Spline Interpolation')
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.title('Plot for Original Image')
	ax.plot_surface(xx, yy, img, color='b')
	plt.show()