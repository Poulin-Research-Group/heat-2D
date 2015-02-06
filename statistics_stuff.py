from collections import OrderedDict
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
plt.clf()

processors = [2, 4]
Ms = range(8, 13)

data_p2 = []
data_p4 = []
data_parallel = {2: data_p2, 4: data_p4}


class StatsDict(OrderedDict):
  """prettified ordered dictionary with statistics about some data"""
  def __init__(self, data):
    super(StatsDict, self).__init__()
    self.data = data
    self.mean = np.mean(data)
    self.std  = np.std(data)
    self.min  = np.min(data)
    self.max  = np.max(data)
    self.median = np.percentile(data, 50)
    self.lower_q = np.percentile(data, 25)
    self.upper_q = np.percentile(data, 75)

    self['mean'] = self.mean
    self['std']  = self.std
    self['min']  = self.min
    self['lower_q'] = self.lower_q
    self['median']  = self.median
    self['upper_q'] = self.upper_q
    self['max']  = self.max

  def __str__(self):
    s = []
    for key in self.keys():
      s.append('%s%s%s' % (key, (15 - len(key)) * ' ', self[key]))  # hacky but whatever
    return '\n'.join(s)


for p in processors:
  for M in Ms:
    filename = './tests/par-step/p%d-M%s.txt' % (p, str(M).zfill(2))
    with open(filename) as f:
      data_parallel[p].append([float(i) for i in f.readlines()])

for i in xrange(len(data_p2)):
  plt.clf()
  d_p2 = data_p2[i]
  stats_p2 = StatsDict(d_p2)

  keys = stats_p2.keys()
  for j in xrange(len(keys)):
    key = keys[j]
    plt.annotate('%.2f' % stats_p2[key], xy=(240, j))

  plt.hist(d_p2)

  d_p4 = data_p4[i]
  stats_p4 = StatsDict(d_p4)

plt.savefig('./pics/p2-data.pdf')
# plt.show()

serial_data = []

for M in Ms:
  filename = './tests/ser-step/M%s.txt' % (str(M).zfill(2))

  with open(filename) as f:
    serial_data.append([float(i) for i in f.readlines()])
