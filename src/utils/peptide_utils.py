import numpy as np

from pyteomics import mzxml, mgf, mass

massH = 1.00782503207;
massO = 15.99491461956;
massH2O = 18.0105646837036;

massDeltK = 8.0141988016;
massPhospho = 79.96633052075;


chem_masses = { 'C':12.0000000, 'H':massH, 'N':14.0030732,'O':massO, 'S':27.976927 };
averagine = { 'C':4.9384, 'H':7.7583, 'N':1.3577, 'O':1.4773, 'S':0.0417 };
averagine_mass = np.sum( [ averagine[e]*chem_masses[e] for e in averagine ] );

def get_elemental_composition( mz, charge ):
    residuemass = (mz - massH) * charge * 1.0;
    aunits = residuemass / averagine_mass;
    acomp = {a: int(round(averagine[a] * aunits)) for a in averagine};

    amass = np.sum([acomp[a] * chem_masses[a] for a in acomp])
    numH = int((residuemass - amass) / massH);
    acomp['H'] += numH;
    amass += numH * massH;

    compstring = '';
    for c in ['C', 'H', 'N', 'O', 'S']: compstring += c+str(acomp[c]);
    compstring += '\t' + str(charge);

    return acomp, amass, compstring;


def getModDict(Modstring=""):
    if not len(Modstring):
        return {}
    ModDict = {
        "A": 0.0,
        "R": 0.0,
        "N": 0.0,
        "D": 0.0,
        "C": 0.0,
        "E": 0.0,
        "Q": 0.0,
        "G": 0.0,
        "H": 0.0,
        "I": 0.0,
        "L": 0.0,
        "K": 0.0,
        "M": 0.0,
        "F": 0.0,
        "P": 0.0,
        "S": 0.0,
        "T": 0.0,
        "W": 0.0,
        "Y": 0.0,
        "V": 0.0,
    }
    ss = Modstring.split(",")
    for s in ss:
        assert s[0] in ModDict
        ModDict[s[0]] += float(s[1:])
    return ModDict


def getAnimoMassDict():
    amino_mono = {
        "A": 71.03711378471,
        "R": 156.1011110236,
        "N": 114.04292744114,
        "D": 115.02694302383,
        "C": 103.00918478470999 + 57.02146,
        "E": 129.04259308797,
        "Q": 128.05857750528,
        "G": 57.02146372057,
        "H": 137.05891185845,
        "I": 113.08406397713,
        "L": 113.08406397713,
        "K": 128.094963014,
        "M": 131.04048491299,
        "F": 147.06841391299,
        "P": 97.05276384884999,
        "S": 87.03202840427,
        "T": 101.04767846841,
        "W": 186.07931294986,
        "Y": 163.06332853255,
        "V": 99.06841391299,
        "U": 150.95363508471,
        "1": 131.04048491299 + massO,
        "2": 87.03202840427 + massPhospho,
        "3": 101.04767846841 + massPhospho,
        "4": 163.06332853255 + massPhospho,
        "5": 128.094963014 + massDeltK,
    }
    return amino_mono;


def getTransformPeptide(peptide):
    peptide1 = peptide;
    peptide1 = peptide1.replace('M[15.99]', '1');
    peptide1 = peptide1.replace('S[79.97]', '2');
    peptide1 = peptide1.replace('T[79.97]', '3');
    peptide1 = peptide1.replace('Y[79.97]', '4');
    return peptide1;


def getTransformPeptide2(peptide):
    peptide2 = peptide;
    peptide2 = peptide2.replace('M[15.99]', 'oM');
    peptide2 = peptide2.replace('S[79.97]', 'pS');
    peptide2 = peptide2.replace('T[79.97]', 'pT');
    peptide2 = peptide2.replace('Y[79.97]', 'pY');
    return peptide2;


def getTransformAnimoMassDict():
    db = mass.Unimod();
    aa_comp = dict(mass.std_aa_comp);
    aa_comp['o'] = mass.Composition(formula='O');
    aa_comp['p'] = mass.Composition('HPO3');
    # aa_comp['c'] = mass.Composition('C4H9');
    return aa_comp;



def neutral_mass(peptide,aminoMass={}):
    if not len(aminoMass): aminoMass = getAnimoMassDict();
    return massH2O+np.sum([aminoMass[i] for i in peptide]);


def fragmentIon(peptide, charge, aminoMass={}):
    if not len(aminoMass): aminoMass = getAnimoMassDict();

    if charge <= 1: charge = 2;
    bion = {}; yion = {};
    bion_mass = [aminoMass[peptide[0]]];
    yion_mass = [aminoMass[peptide[-1]] + massH2O];

    for p in peptide[1:-1]:
        bion_mass.append(bion_mass[-1]+aminoMass[p])
    # print peptide[::-1][1:-1]
    for p in peptide[::-1][1:-1]:
        yion_mass.append(yion_mass[-1]+aminoMass[p])
    for c in range(1, charge):
        bion[c] = [(b+c*massH)/c for b in bion_mass]
        yion[c] = [(y+c*massH)/c for y in yion_mass]
    return bion, yion;


# def fragmentIonMass(peptide, aminoMass):
#     bion = {}; yion = {};
#     bion_mass = [aminoMass[peptide[0]]]
#     yion_mass = [aminoMass[peptide[-1]] + 2*massH + massO]
#     for p in peptide[1:-1]:
#         bion_mass.append(bion_mass[-1]+aminoMass[p])
#     # print peptide[::-1][1:-1]
#     for p in peptide[::-1][1:-1]:
#         yion_mass.append(yion_mass[-1]+aminoMass[p])
#     return np.concatenate((np.array(bion_mass,dtype=np.float32), np.array(yion_mass,dtype=np.float32)))
#


def get_unmod_peptide(peptide):
    peptide2 = peptide; mod_list = [];
    while True:
        start_idx = peptide2.find('[');
        if start_idx < 0: break;
        end_idx = peptide2.find(']'); assert end_idx > 0;

        mod_list.append((start_idx-1, peptide2[start_idx:end_idx+1] ));
        peptide2 = peptide2[:start_idx] + peptide2[end_idx+1:];

    return peptide2, mod_list;



