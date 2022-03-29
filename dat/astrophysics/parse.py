from pdb import set_trace as st
import os

from glob import glob


buf = []
files = sorted(glob('*.txt'))

for fil in files:
    with open(fil, 'r') as ff:
        ff = ff.read()
        ff = ff.split('\n')

        for ll in ff:
            if len(ll) > 28:
                ll = ll.encode('ascii', 'ignore').decode('utf-8')
                ll = ll.strip('-')
                ll = ll.replace('\x0c', ' ')
                buf.append(ll)

buf = ' '.join(buf)

with open('astrophysics.txt', 'w') as ff:
    ff.write(buf)

st()
