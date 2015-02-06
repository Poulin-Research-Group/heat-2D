# remove all content from files

import subprocess
d = 'ser-step'
for M in xrange(8, 13):
    D1 = './%s/M%s.txt' % (d, str(M).zfill(2))
    D2 = './%s/solution-M%s.txt' % (d, str(M).zfill(2))
    subprocess.call('> ' + D1, shell=True)
    subprocess.call('> ' + D2, shell=True)

d = 'par-step'
for p in xrange(2, 6, 2):
    for M in xrange(8, 13):
        D1 = './%s/p%d-M%s.txt' % (d, p, str(M).zfill(2))
        D2 = './%s/solution-p%d.txt' % (d, p)
        subprocess.call('> ' + D1, shell=True)
        subprocess.call('> ' + D2, shell=True)
