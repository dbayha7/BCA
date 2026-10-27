import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8')); b=json.load(open(sys.argv[2],encoding='utf-8'))
sr=b['datasetidentity']['split_receipt']
print('times equal', sr['calibration_times']==r['calibration_times'])
print('rowids equal', sr['calibration_row_ids']==r['row_ids']['calibration'])
print('scale idx equal', sr['scale_bank_training_row_indices']==r['fit_training_row_indices'])
print('sel/final used flag', sr['selection_and_final_calibration_used'])
print('normalization keys', list(b['normalization'].keys()) if isinstance(b['normalization'],dict) else type(b['normalization']))
