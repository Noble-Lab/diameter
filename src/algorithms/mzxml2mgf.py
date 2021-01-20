import os, sys, random, re, gc;
import numpy as np
import pandas as pd

from pyteomics import mzxml,mgf


def convertMzxml2mgf(MzxmlFilename, outputMgfFilename, charge):
    assert os.path.isfile(MzxmlFilename);

    mgfList = [];
    for row in mzxml.read(MzxmlFilename):
        if int(row['msLevel']) == 2:
            mgfEntry = {
                'm/z array': row['m/z array'],
                'intensity array': row['intensity array'],
                'params': {
                    'title': row['id'],
                    'pepmass': row["precursorMz"][0]["precursorMz"],
                    'rtinseconds': row["retentionTime"],
                    'charge': "%d+" % charge,
                    'scans': int(row['id'])
                }
            }
            # assert 10*int(row['id'])+1 not in scan_number_set
            # scan_number_set.add(10*int(row['id'])+1)
            mgfList.append(mgfEntry)

    print(outputMgfFilename);
    mgf.write(spectra=mgfList, output=outputMgfFilename, file_mode='w')


def main():
    MzxmlFilename = sys.argv[1];
    outputMgfFilename = sys.argv[2];
    charge = int(sys.argv[3]);
    convertMzxml2mgf(MzxmlFilename, outputMgfFilename, charge);


if __name__ == '__main__':
    main();
