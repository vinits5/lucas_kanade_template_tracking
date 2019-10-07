# Lucas Kanade Template Tracker

### Results on Car Sequence
![Car Result](https://github.com/vinits5/lucas_kanade_template_tracking/blob/master/result/output.gif)

### How to use code?
```python
frames = np.load(path)
```
Provide the dataset "path"!

`python test.py`

1. Chose Bounding Box for the area to be tracked.
2. Press Space/Enter.

### What is RectBivariateSpline.
It is a spline interpolation to evaluate the image at continuous locations. An approximate spline curve is fitted with (x,y) as the input and pixel intensity at (x,y) as the output.

Some visual results:
<p float="center">
	<img src="https://github.com/vinits5/lucas_kanade_template_tracking/blob/master/result/original_image.jpg" width="640" height="480" />
	<img src="https://github.com/vinits5/lucas_kanade_template_tracking/blob/master/result/interpolated_image.jpg" width="640" height="480" /> 
</p>