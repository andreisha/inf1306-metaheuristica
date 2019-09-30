#!/bin/bash
#python gvrp.py
#files = {'GVRP/A-n32-k5-C11-V2.gvrp', 'GVRP/A-n32-k5-C11-V2.gvrp'}

#for file in $files
#for file in 'GVRP/A-n32-k5-C11-V2.gvrp' 'GVRP/A-n32-k5-C11-V2.gvrp'

cd GVRP
files=`ls`
for file in $files
do
    python gvrp.py "${file}" >> output.txt
done