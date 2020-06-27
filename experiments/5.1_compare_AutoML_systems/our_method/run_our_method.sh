export OMP_NUM_THREADS=1

cat combination_indices.csv | xargs -i --max-procs=50 bash -c "python our_method_evaluation_all.py {}"