#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import glob
import numpy as np
from scipy.optimize import curve_fit
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-s','--survey',default='ps1',choices=['ps1','des'])
args = parser.parse_args()
survey = args.survey

def func(x,a,b,c):
    return a/(x - b) + c

# Initial guess
P0 = [[20.0,  7.0, 4.0],
      [20.0,  4.0, 4.0 ],
      [15.0,  1.0, 4.0],
      [10.0, -1.0, 4.0],
      [8.0,  -2.0, 4.0],
      [4.0,  -3.0, 4.0],
     ]
BOUNDS = [[0, -10, 3.5],
          [30, 10, 4.5]]

files = sorted(glob.glob('%s_p50_d*.npy'%survey))
results = []

for i,f in enumerate(files):
    dist = float(f.rsplit('_')[-1].rsplit('.',1)[0].strip('d'))
    data = np.load(f)
    sigma = 0.05*np.ones(len(data[:,1]))
    sigma[0] = 0.5
    sigma[-1] = 0.5
    #print sigma
    r = curve_fit(func,data[:,0],data[:,1],p0=P0[i],sigma=sigma,bounds=BOUNDS)
    results += [[dist]+r[0].tolist()]

results = np.asarray(results)
results = results[np.argsort(results[:,0])]

print '%-5s  %-5s  %-5s  %-5s'%('Dist','A0','Mv0','logr0')
for r in results:
    print '[%-5.1f,  %-5.1f,  %-5.1f,  %-5.1f],'%tuple(r)

