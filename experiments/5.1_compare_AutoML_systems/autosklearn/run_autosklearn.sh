export OMP_NUM_THREADS=1

cat combination_indices.csv | xargs -i --max-procs=20 bash -c "python autosklearn_evaluation_all.py {}"