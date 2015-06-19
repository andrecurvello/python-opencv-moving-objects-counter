# Introduction #

This is a description of the computer vision/image analysis/artificial intelligence algorithm for counting moving objects in video streams.

# Operational algorithm #

  1. Image acquisition (webcam or other video camera)
  1. Image pre-processing (scaling, cropping, smoothing, color->gray?)
  1. Image segmentation (foreground/background, background extraction, image differencing)
  1. Object detection (connected component labelling, mathematical morphology)
  1. Object tracking (connected component tracking, optical flow, contour matching)
  1. Object classification (connected component statistics (size, shape), tracking statistics (speed)
  1. Object counting/reporting

# Training algorithm #

  1. Run the above algorithm on training set and find parameters
  1. Run the algorithm on test set and compute skill scores (POD, FAR, etc.)
  1. Repeat this to find best parameters