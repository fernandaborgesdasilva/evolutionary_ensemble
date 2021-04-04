#!/bin/bash
export OPENBLAS_NUM_THREADS=1
input_file=breast
num_classifiers=10
#num_iterations=19448
num_iterations=10
mkdir /home-ext/fbslnlv/commit_bc8adbd/$input_file
pushd /home-ext/fbslnlv/commit_bc8adbd/
pushd $input_file
echo "1 >>>>>>>>> diversity_ensemble.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/diversity_ensemble.py -i $input_file -o saida_dce -e $num_classifiers -s $num_iterations &> output_dce_$input_file.txt
echo "\n2 >>>>>>>>> brute_force_search_exec.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/brute_force_search_exec.py -i $input_file -o saida_bf -e $num_classifiers -s $num_iterations &> output_bf_$input_file.txt
echo "\n3 >>>>>>>>> brute_force_random_search_exec.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/brute_force_random_search_exec.py -i $input_file -o saida_bfr -e $num_classifiers -s $num_iterations &> output_bfr_$input_file.txt
echo "\n4 >>>>>>>>> diversity_grid.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/diversity_grid.py -i $input_file -o saida_dce_grid -e $num_classifiers -s $num_iterations &> output_dce_grid_$input_file.txt
echo "\n5 >>>>>>>>> random_search.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/random_search.py -i $input_file -o saida_rs -e $num_classifiers -s $num_iterations &> output_rs_$input_file.txt
iteration=6
#list='2 4 8 16 32'
list='2 4 8'
for num_cores in $list;do 
    echo "\n$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o saida_pdce -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_brute_force_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_search_exec.py -i $input_file -o saida_pbf -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbf_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_brute_force_random_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_random_search_exec.py -i $input_file -o saida_pbfr -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbfr_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o saida_pdce_grid -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_grid_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_0 -e $num_classifiers -s $num_iterations -c $num_cores -p 0 &> output_prs_0_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_1 -e $num_classifiers -s $num_iterations -c $num_cores -p 1 &> output_prs_1_$input_file.txt
    let iteration=$iteration+1
done
list='16 32'
for num_cores in $list;do 
    echo "\n$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o saida_pdce -e $num_cores -s $num_iterations -c $num_cores &> output_pdce_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_brute_force_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_search_exec.py -i $input_file -o saida_pbf -e $num_cores -s $num_iterations -c $num_cores &> output_pbf_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_brute_force_random_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_random_search_exec.py -i $input_file -o saida_pbfr -e $num_cores -s $num_iterations -c $num_cores &> output_pbfr_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o saida_pdce_grid -e $num_cores -s $num_iterations -c $num_cores &> output_pdce_grid_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_0 -e $num_cores -s $num_iterations -c $num_cores -p 0 &> output_prs_0_$input_file.txt
    echo "\n$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_1 -e $num_cores -s $num_iterations -c $num_cores -p 1 &> output_prs_1_$input_file.txt
    let iteration=$iteration+1
done
popd
popd