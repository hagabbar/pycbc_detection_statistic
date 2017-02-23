# pycbc_ml_working

Instructions on how to run: <br />

Required Dependencies: keras, tensorflow, matplotlib, numpy, h5py, pycbc, scipy, sympy

If you want to use just one GPU for a run then you must set the following environment variable to your desired GPU number <br />
export CUDA_VISIBLE_DEVICES="1"

1.) Login to lho cluster. <br />
    gsissh albert.einstein@ldas-pcdev2.ligo-wa.caltech.edu
    
2.) Login to dgx1 machine. (you must ask Stuart for an account fist ... anderson@ligo.caltech.edu) <br />
    ssh dgx-1 <br />
    
3.) In feature\_gen\_scripts, run get\_sngl\_stats python script to get features for testing and training. <br />
e.g. python get\_sngl\_stats --ifo H1 --single-trigger-files H1-HDF\_TRIGGER\_MERGE\_FULL\_DATA-1128299417-1083600.hdf --veto-file H1L1-CUMULATIVE\_CAT\_12H\_VETO\_SEGMENTS.xml --veto-segment-name CUMULATIVE\_CAT\_12H --found-injection-file H1L1-HDFINJFIND\_BBH01\_INJ\_INJ\_INJ-1128299417-1083600.hdf --window 1 --output-file BBH01\_test.hdf --temp-bank H1L1-BANK2HDF-1128299417-1083600.hdf --inj-file H1-HDF\_TRIGGER\_MERGE\_BBH01\_INJ-1128299417-1083600.hdf --inj-coinc-file H1L1-HDFINJFIND\_BBH01\_INJ\_INJ\_INJ-1128299417-1083600.hdf --ifar-thresh 0.1 --verbose <br />


4.) In main directory, run pycbc_neural_network.py script to train/test on feature set generated in the previous step. <br />
e.g. python pycbc_neural_network.py -d path/to/data/chunk*/*.hdf -b path/to/one/result/file/from/each/chunk\*/BBH01.hdf -o /path/to/output/directory -t 0.7 -e 10 -bs 1000 -u usertag  >/dev/null 2>err.txt & <br />
