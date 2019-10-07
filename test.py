import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from LucasKanade import *

def drawRectangle(img, xi, yi, size_x, size_y):
	return cv2.rectangle(img, (yi, xi), (yi+size_y, xi+size_x), (0,0,255), 2)

def find_bb(img):
	x,y,w,h = list(cv2.selectROI(img))
	rect = [x, y, x+w, y+h]
	return rect
	
def main(frames, rect):
	print('Started Tracking!')
	width = abs(rect[3] - rect[1])
	length = abs(rect[2] - rect[0])

	for i in range(frames.shape[2]-1):
		print('Frame Idx: ',i)
		# rectList.append(rect.copy())
		It = frames[:,:,i]
		It1 = frames[:,:,i+1]
		p = LucasKanade(It, It1, rect)
		rect[0] += p[0]
		rect[1] += p[1]
		rect[2] += p[0]
		rect[3] += p[1]

		img = drawRectangle(frames[:,:,i], int(rect[1]), int(rect[0]), width, length)

		cv2.imshow('frame', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			  break
	print('End Tracking...!')
	return None

if __name__ == '__main__':
	frames = np.load('data/sylvseq.npy')
	# rect = [59, 116, 145, 151]
	rect = find_bb(frames[:,:,0])
	# rectList = []
	main(frames, rect)

	# np.save('carseqrects.npy',rectList)