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
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/diversity_ensemble.py -i $input_file -o saida_dce -e $num_classifiers -s $num_iterations &> output_dce.txt
echo "2 >>>>>>>>> brute_force_search_exec.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/brute_force_search_exec.py -i $input_file -o saida_bf -e $num_classifiers -s $num_iterations &> output_bf.txt
echo "3 >>>>>>>>> brute_force_random_search_exec.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/brute_force_random_search_exec.py -i $input_file -o saida_bfr -e $num_classifiers -s $num_iterations &> output_bfr.txt
echo "4 >>>>>>>>> diversity_grid.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/diversity_grid.py -i $input_file -o saida_dce_grid -e $num_classifiers -s $num_iterations &> output_dce_grid.txt
echo "5 >>>>>>>>> random_search.py"
time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/random_search.py -i $input_file -o saida_rs -e $num_classifiers -s $num_iterations &> output_rs.txt
iteration=5
#list='2 4 8 16 32'
list='2 4 8'
for num_cores in $list;do 
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o saida_pdce_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_search_exec.py -i $input_file -o saida_pbf_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbf_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_random_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_random_search_exec.py -i $input_file -o saida_pbfr_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbfr_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o saida_pdce_grid_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_grid_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_0_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 0 &> output_prs_0_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_1_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 1 &> output_prs_1_$num_cores.txt
done
list='16 32'
for num_cores in $list;do 
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o saida_pdce_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_search_exec.py -i $input_file -o saida_pbf_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbf_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_random_search_exec.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_brute_force_random_search_exec.py -i $input_file -o saida_pbfr_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbfr_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o saida_pdce_grid_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_grid_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_0_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 0 &> output_prs_0_$num_cores.txt
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python -u /home-ext/fbslnlv/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_1_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 1 &> output_prs_1_$num_cores.txt
done
popd
popd