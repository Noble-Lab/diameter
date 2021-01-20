import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import os
import pylab;
import numpy as np

def plot(results, outputfile):
    linestyle = ["o", "v", "^", ">", "<", "1", "2", "3", "4", "s", "p", "*"]
    # assert len(results) <= len(linestyle)
    matplotlib.use('Agg')
    legend = []
    targets = []
    decoys = []
    minl = 100000000

    for f, l in results:
        legend.append(l)
        with open(f) as input:
            print f
            target = []
            decoy = []
            t, d, n = 0, 0, 0

            for row in csv.DictReader(input,delimiter='\t'):
                if int(row["charge"]) in [1, 2, 3, 4, 5]:

                    if 't' == row['td']:
                        t += 1
                    elif 'd' == row['td']:
                        d += 1
                    else:
                        assert False
                    target.append(t)
                    decoy.append(d)
                    # if d >= 400:
                #     break
            targets.append(target)
            decoys.append(decoy)
            minl = min(minl, decoy[-1], 10000)
    maxl = 0

    for target, decoy in zip(targets, decoys):
        i = 0
        q = 0.1
        for j in range(len(decoy)):
            if decoy[j]/(float(target[j])+ np.finfo(np.float).eps) < q:
                i = decoy[j]
            else:
                break
        maxl = max(i, maxl)
    print maxl
    minl = min(maxl, minl)
    counter = 0
    for target, decoy in zip(targets, decoys):
        i = 0
        q = 0.0
        1
        for j in range(len(decoy)):
            if decoy[i] <= minl:
                i+=1
            else:
                break
        i = min(i, len(decoy))
        plt.plot(decoy[:i], target[:i], marker=linestyle[counter % len(linestyle)],markevery=int(i/10))
        counter += 1
    plt.legend(legend, loc='center right')
    plt.xlabel("selected decoy peptides", fontsize=18)
    plt.ylabel("selected target peptides", fontsize=18)
    outputFloder = os.path.dirname(outputfile)
    if not os.path.exists(outputFloder):
        os.makedirs(outputFloder)
    print "save fig as %s"%outputfile

    F = pylab.gcf()
    DPI = F.get_dpi()
    DefaultSize = F.get_size_inches()
    F.set_size_inches((DefaultSize[0] * 2, DefaultSize[1] * 2))

    plt.savefig(outputfile)
    plt.clf()

def sanityCheck(results):
    target = {}
    decoy = {}
    target_decoy = {}
    print results
    for f, _ in results:
        with open(f) as input:
            target[f] = set()
            decoy[f] = set()
            target_decoy[f] = []

            for row in csv.DictReader(input,delimiter='\t'):
                if row['td'] == 't':
                    assert (row['peptide'], row['charge']) not in target[f]
                    target[f].add((row['peptide'], row['charge']))
                if row['td'] == 'd':
                    assert (row['peptide'], row['charge']) not in decoy[f]
                    decoy[f].add((row['peptide'], row['charge']))
                target_decoy[f].append((row['td'],(row['peptide'], row['charge'])))
            print "%s contains %d targets and %d decoys"%(f, len(target[f]),  len(decoy[f]))
    high_target = {}
    for f, _ in results:
        t = 0
        d = 0
        q = 0.01
        j = 0
        for i in target_decoy[f]:
            if i[0] == 't':
                t += 1
            else:
                d += 1
            j += 1
            if d/(t+0.00001)> q:
                break
        j = min(j, len(target_decoy[f]))
        high_target[f] = [i[1] for i in target_decoy[f][:j] if i[0] == 't']


    for f, _ in results:
        print "===========\n" + f
        print "target :%d"%len(target[f])
        print "high target %d"%len(high_target[f])
        print "decoy :%d"%len(decoy[f])

        for g, _ in results:
            if f == g:
                continue
            print "==%s\n"%(g)
            count_in = 0
            count_notin = 0
            for t in high_target[f]:
                assert (t[0], '1') not in decoy[g]
                assert (t[0], '2') not in decoy[g]
                assert (t[0], '3') not in decoy[g]
                if t in target[g]:
                    count_in += 1
                else:
                    count_notin += 1
            print "in: %d, not in, %d"%(count_in, count_notin)
            for d in decoy[f]:
                assert (d[0], '1') not in target[g]
                assert (d[0], '2') not in target[g]
                assert (d[0], '3') not in target[g]
