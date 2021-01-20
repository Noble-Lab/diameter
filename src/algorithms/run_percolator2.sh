#!/bin/bash
inputUrl=$1
fastaFile=$2
percolatorFolder=$3
cruxExecFile=$4
trainFDR=$5
testFDR=$6
inputUrl2=$7

echo inputUrl $inputUrl
echo fastaFile $fastaFile
echo percolatorFolder $percolatorFolder
echo cruxExecFile $cruxExecFile
echo trainFDR $trainFDR
echo testFDR $testFDR
echo inputUrl2 $inputUrl2

if [ ! -d $percolatorFolder ]; then
   mkdir -p $percolatorFolder
fi

#fdr=0.01
tdc=F

#if [ ! -f $percolatorFolder/percolator.target.psms.txt ]; then
    $cruxExecFile percolator \
    --tdc $tdc \
    --train-fdr $trainFDR \
    --test-fdr $testFDR \
    --output-dir $percolatorFolder \
    --output-weights T \
    --overwrite T \
    --unitnorm T \
    $inputUrl $inputUrl2
#fi

#--protein true \
#--picked-protein $fastaFile \
    
