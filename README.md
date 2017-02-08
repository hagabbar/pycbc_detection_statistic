# pycbc_ml_working

Instructions on how to run:
1.) In feature_gen_scripts, run get_sngl_stats python script to get features for testing and training.
e.g. python get_sngl_stats --ifo H1 --single-trigger-files H1-HDF_TRIGGER_MERGE_FULL_DATA-1128299417-1083600.hdf --veto-file H1L1-CUMULATIVE_CAT_12H_VETO_SEGMENTS.xml --veto-segment-name CUMULATIVE_CAT_12H --found-injection-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --window 1 --output-file BBH01_test.hdf --temp-bank H1L1-BANK2HDF-1128299417-1083600.hdf --inj-file H1-HDF_TRIGGER_MERGE_BBH01_INJ-1128299417-1083600.hdf --inj-coinc-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --ifar-thresh 0.1 --verbose

2.) In main directory, run simple_neural_network.py script to train/test on feature set generated in the previous step.
e.g. python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &
