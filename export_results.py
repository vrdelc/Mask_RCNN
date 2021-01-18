import tensorflow as tf
import time
import csv
import sys
import os
import collections
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Export loss funcions.')
parser.add_argument('--logs', required=True,
                    metavar="/path/to/logs/",
                    help='Directory of the logs')
parser.add_argument('--file', required=False,
                    default='metrics.csv',
                    metavar="<metrics_file>",
                    help='Metric file (csv)')
args = parser.parse_args()

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False;
# TF version < 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.python.summary import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version >= 1.3.0
if (not eventAccumulatorImported):
	try:
		from tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = Unknown
if (not eventAccumulatorImported):
	raise ImportError('Could not locate and import Tensorflow event accumulator.')

class Timer(object):
	# Source: https://stackoverflow.com/a/5849861
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print('[%s]' % self.name)
			print('Elapsed: %s' % (time.time() - self.tstart))

with Timer():
	ea = event_accumulator.EventAccumulator(args.logs,
  	size_guidance={
      	event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
      	event_accumulator.IMAGES: 0,
      	event_accumulator.AUDIO: 0,
      	event_accumulator.SCALARS: 0,
      	event_accumulator.HISTOGRAMS: 0,
	})

with Timer():
	ea.Reload() # loads events from file

tags = ea.Tags();
for t in tags:
	tagSum = []
	if (isinstance(tags[t],collections.Sequence)):
		tagSum = str(len(tags[t])) + ' summaries';
	else:
		tagSum = str(tags[t]);

scalarTags = tags['scalars'];
with Timer():
	with open(args.file,'w') as csvfile:
		logWriter = csv.writer(csvfile, delimiter=',');

		# Write headers to columns
		headers = ['wall_time','step'];
		for s in scalarTags:
			headers.append(s);
		logWriter.writerow(headers);

		vals = ea.Scalars(scalarTags[0]);
		for i in range(len(vals)):
			v = vals[i];
			data = [v.wall_time, v.step];
			for s in scalarTags:
				scalarTag = ea.Scalars(s);
				S = scalarTag[i];
				data.append(S.value);
			logWriter.writerow(data);
