from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import sys, os

sys.path.append("src")

import matplotlib as mpl
import matplotlib.pyplot as plt

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

from sklearn.externals import joblib

from eig_model import *


def getvelocities(x, y, frame_times, mpp):
	dx = x.diff()[1::]
	dy = y.diff()[1::]
	d = (dx.rpow(2) + dy.rpow(2)).rpow(1/2)*mpp # in micrometers
	ft = DataFrame(frame_times.repeat(x.shape[1], axis=1))
	dt = ft.diff()[1::] / 1000 # convert to seconds
	return d.div(dt)

def nextdataset(frames):
	frames.default_coords['m'] += 1

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print('Please input your file: ', end='')
		fn = input()
		if not fn:
			cdir, cfn = os.path.split(os.path.abspath(__file__))
			fn = "E:\\workspace\\matlab\\MDA\\MDA-MB-231\\MDA-MB-231 2.5 mg per ml.nd2"
	else:
		fn = sys.argv[1]
	frames = pims.open(fn)

	# set to DIC, which is in the second c layer
	frames.default_coords['c'] = 1
	mpp = frames.metadata['calibration_um']

	CELL_SIZE_PX = 21
	SZ = (CELL_SIZE_PX, CELL_SIZE_PX)
	MIN_MASS = 7300

	model = model_build(sz=SZ)

	for i in range(0, frames.sizes['m']):
		frames.default_coords['m'] = i

		frame_times = np.array([fr.metadata['t_ms'] for fr in frames], ndmin=2).transpose()

		if len(sys.argv) < 3:
			f = tp.batch(frames[:], CELL_SIZE_PX, minmass=MIN_MASS)
		else:
			if os.path.isfile(sys.argv[2]):
				f = joblib.load(sys.argv[2])
			else:
				f = tp.batch(frames[:], CELL_SIZE_PX, minmass=MIN_MASS)
				joblib.save(sys.argv[2], f, compress=9)

		t = tp.link_df(f, 20, memory=3)

		x = t.set_index(['frame', 'particle'])['x'].unstack()
		y = t.set_index(['frame', 'particle'])['y'].unstack()

		distances = find_feature_similarity(model, frames, x, y, SZ)

		ax = plt.gca()
		ax.scatter(x, y, 5, distances / np.nanmax(distances), cmap='coolwarm')

		# v = getvelocities(x, y, frame_times, mpp)
		# c = v / v.max().max()
		# plt.figure()
		# ax = plt.gca()
		# ax.scatter(x[1::],y[1::], 5, c=c, cmap='coolwarm')

		ax.invert_yaxis()
		plt.show()

	frames.close()
