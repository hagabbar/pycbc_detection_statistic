# pycbc_ml_working

Instructions on how to run: <br />
1.) In feature\_gen\_scripts, run get\_sngl\_stats python script to get features for testing and training. <br />
e.g. python get\_sngl\_stats --ifo H1 --single-trigger-files H1-HDF\_TRIGGER\_MERGE\_FULL\_DATA-1128299417-1083600.hdf --veto-file H1L1-CUMULATIVE\_CAT\_12H\_VETO\_SEGMENTS.xml --veto-segment-name CUMULATIVE\_CAT\_12H --found-injection-file H1L1-HDFINJFIND\_BBH01\_INJ\_INJ\_INJ-1128299417-1083600.hdf --window 1 --output-file BBH01\_test.hdf --temp-bank H1L1-BANK2HDF-1128299417-1083600.hdf --inj-file H1-HDF\_TRIGGER\_MERGE\_BBH01\_INJ-1128299417-1083600.hdf --inj-coinc-file H1L1-HDFINJFIND\_BBH01\_INJ\_INJ\_INJ-1128299417-1083600.hdf --ifar-thresh 0.1 --verbose <br />

2.) In main directory, run simple\_neural\_network.py script to train/test on feature set generated in the previous step. <br />
e.g. python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt & <br />
