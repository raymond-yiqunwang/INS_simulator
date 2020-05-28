#/bin/bash

for itrial in {101..200}
do
    dirname='trial-'$itrial
    mkdir $dirname
    cp job_submit.sh $dirname
    cp SPOSCAR $dirname
    cp FORCE_SETS $dirname
    cp DSF_copper.py $dirname
    cp clean.sh $dirname
    cd $dirname
        echo $PWD
        qsub job_submit.sh
        cd ..
done
