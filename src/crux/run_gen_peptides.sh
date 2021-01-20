#!/bin/bash
cruxExecFile=$1
fastaFile=$2
cleavages=$3
enzyme=$4
modsSpec=$5

echo cruxExecFile $cruxExecFile
echo fastaFile $fastaFile
echo cleavages $cleavages
echo enzyme $enzyme
echo modsSpec $modsSpec

extension=.${fastaFile#*.}
databaseFile=${fastaFile%$extension}-shuffle
echo databaseFile $databaseFile

if [ ! -d $databaseFile ]; then
    $cruxExecFile generate-peptides \
    --decoy-format shuffle \
    --overwrite F \
    --enzyme $enzyme \
    --missed-cleavages 0 \
    --output-dir $databaseFile \
    $fastaFile
    
    mv ${databaseFile}/generate-peptides.proteins.decoy.txt $databaseFile/..
fi

databaseFile=${fastaFile%$extension}-origin
echo databaseFile $databaseFile
if [ ! -d $databaseFile ]; then
    $cruxExecFile tide-index \
    --peptide-list T \
    --decoy-format peptide-reverse \
    --overwrite F \
    --missed-cleavages $cleavages \
    --mods-spec $modsSpec \
    --enzyme $enzyme \
    --max-mass 6000 \
    --output-dir $databaseFile \
    $fastaFile $databaseFile
fi


