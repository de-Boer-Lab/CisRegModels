#!/bin/bash
set -e #make it so that the script exits completely if one command fails
echo "############################################"
echo "Creating One-Hot representation of sequences"
echo "############################################"
seqsToOHC.py -i example/HighQuality.pTpA.Glu.test.txt.gz -m 110 -o example/HighQuality.pTpA.Glu.test.OHC.gz

echo "############################################"
echo "learning model from known motifs, holding concentrations and motifs static"
echo "############################################"
makeThermodynamicEnhancosomeModel.py -i example/HighQuality.pTpA.Glu.test.OHC.gz  -o example/test.model.AP -eb -sl 110 -nm 250 -ml 30  -b 128 -v -v -v -se 100000 -dm example/allTF_PKdMFiles_polyA_and_FZF1.txt  -ic example/allTF_minKds_polyA_and_FZF1.txt -po -lr 0.04 -ntm -ntc -r 20 -ia 0.01 -ip 0.01

echo "############################################"
echo "now learning additional parameters: concentration and motif"
echo "############################################"
makeThermodynamicEnhancosomeModel.py -i example/HighQuality.pTpA.Glu.test.OHC.gz  -o example/test.model.APCM -eb -sl 110 -nm 250 -ml 30  -b 128 -v -v -v -se 100000 -dm example/allTF_PKdMFiles_polyA_and_FZF1.txt  -ic example/allTF_minKds_polyA_and_FZF1.txt -po -lr 0.001  -r 20 -ia 0.01 -ip 0.01 -res example/test.model.AP.ckpt

echo "############################################"
echo "now learning additional parameters: positional activities"
echo "############################################"
makeThermodynamicEnhancosomeModel.py -i example/HighQuality.pTpA.Glu.test.OHC.gz  -o example/test.model.APCM.pos -eb -sl 110 -nm 250 -ml 30  -b 128 -v -v -v -se 100000 -dm example/allTF_PKdMFiles_polyA_and_FZF1.txt  -ic example/allTF_minKds_polyA_and_FZF1.txt -po -lr 0.0005  -r 50 -ia 0.01 -ip 0.01 -res example/test.model.APCM.ckpt -posa -stra

echo "############################################"
echo "testing final model on training data (i.e. training-test violation)"
echo "############################################"
predictThermodynamicEnhancosomeModel.py -i example/HighQuality.pTpA.Glu.test.OHC.gz -sl 110 -b 128 -M ./example/test.model.APCM.pos.Model -o example/test.model.APCM.pos.pred.gz
echo "See example/test.model.APCM.pos.pred.gz for predictions"

echo "############################################"
echo "testing ACPM  model on training data and outputing predicted binding"
echo "############################################"
predictThermodynamicEnhancosomeModel.py -i example/HighQuality.pTpA.Glu.test.OHC.gz -sl 110 -b 128 -M ./example/test.model.APCM.Model -o example/test.model.APCM.pred.gz -ob
echo "See example/test.model.APCM.pred.gz for predictions"
echo "ALL COMMANDS SUCCEEDED"
