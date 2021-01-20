#!/bin/bash
cruxExecFile=$1
fastaFile=$2
precursorWindow=$3
mzBinWidth=$4
mzBinOffset=$5
cleavages=$6
topMatch=$7
charge=$8
modsSpec=$9
enzyme=${10}
inputFile=${11}
outputfolder=${12}

echo cruxExecFile $cruxExecFile
echo fastaFile $fastaFile
echo precursorWindow $precursorWindow
echo mzBinWidth $mzBinWidth
echo mzBinOffset $mzBinOffset
echo cleavages $cleavages
echo topMatch $topMatch
echo charge $charge
echo modsSpec $modsSpec
echo enzyme $enzyme
echo inputFile $inputFile
echo outputfolder $outputfolder
#mkdir -p $outputfolder

extension=.${fastaFile#*.}
databaseFile=${fastaFile%$extension}-index
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

# --use-tailor-calibration T \
if [ ! -f $outputfolder/tide-search.txt ]; then
    $cruxExecFile tide-search \
    --precursor-window $precursorWindow \
    --precursor-window-type mz \
    --mz-bin-width $mzBinWidth \
    --mz-bin-offset $mzBinOffset \
    --concat T \
    --top-match $topMatch \
    --exact-p-value F \
    --output-dir $outputfolder \
    --overwrite T \
    --num-threads 1 \
    $inputFile $databaseFile
fi




