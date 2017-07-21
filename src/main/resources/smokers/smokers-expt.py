import sys
import time

from tensorlog import comline
from tensorlog import program
from tensorlog import expt
from tensorlog import declare
from tensorlog import mutil

if __name__=="__main__":

    optdict,args = comline.parseCommandLine('--prog smokers.ppr --proppr --db smokers.cfacts'.split())
    ti = program.Interp(optdict['prog'])
    ti.prog.setRuleWeights()
    ti.prog.maxDepth = 99
    rows = []
    for line in open('query-entities.txt'):
        sym = line.strip()
        rows.append(ti.db.onehot(sym))
    X = mutil.stack(rows)

    start0 = time.time()
    for modeString in ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]:
        print 'eval',modeString,
        start = time.time()
        ti.prog.eval(declare.asMode(modeString), [X])
        print 'time',time.time() - start,'sec'
    print 'total time', time.time() - start0,'sec'
