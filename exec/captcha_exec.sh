#!/bin/bash
#SBATCH -J breast
#SBATCH -o breast.%j.out
#SBATCH -e breast.%j.err
#SBATCH -n 32
#SBATCH --mem=20GB
#SBATCH -w compute-0-3
export OPENBLAS_NUM_THREADS=1
input_file=breast
num_classifiers=10
num_iterations=19448
source /home/covoes/env_v_p/bin/activate
mkdir /home/covoes/fernanda/commit_8d71876/$input_file
#rm -rf /tmp/fernanda/commit_8d71876/input_file
mkdir /tmp/fernanda/commit_8d71876/$input_file
pushd /tmp/fernanda/commit_8d71876/
pushd breast
echo "1 >>>>>>>>> diversity_ensemble.py"
time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/diversity_ensemble.py -i $input_file -o saida_dce -e $num_classifiers -s $num_iterations &> output_dce.txt
cp /tmp/fernanda/commit_8d71876/$input_file/saida_dce /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_dce
cp /tmp/fernanda/commit_8d71876/$input_file/output_dce.txt /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_dce.txt
cp /tmp/fernanda/commit_8d71876/$input_file/diversity_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/diversity_fold_*
echo "2 >>>>>>>>> brute_force_search_exec.py"
time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/brute_force_search_exec.py -i $input_file -o saida_bf -e $num_classifiers -s $num_iterations &> output_bf.txt
cp /tmp/fernanda/commit_8d71876/$input_file/saida_bf /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_bf
cp /tmp/fernanda/commit_8d71876/$input_file/output_bf.txt /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_bf.txt
cp /tmp/fernanda/commit_8d71876/$input_file/bfec_seq_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/bfec_seq_fold_*
echo "3 >>>>>>>>> brute_force_random_search_exec.py"
time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/brute_force_random_search_exec.py -i $input_file -o saida_bfr -e $num_classifiers -s $num_iterations &> output_bfr.txt
cp /tmp/fernanda/commit_8d71876/$input_file/saida_bfr /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_bfr
cp /tmp/fernanda/commit_8d71876/$input_file/output_bfr.txt /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_bfr.txt
cp /tmp/fernanda/commit_8d71876/$input_file/bfec_rand_results_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/bfec_rand_results_fold_*
echo "4 >>>>>>>>> diversity_grid.py"
time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/diversity_grid.py -i $input_file -o saida_dce_grid -e $num_classifiers -s $num_iterations &> output_dce_grid.txt
cp /tmp/fernanda/commit_8d71876/$input_file/saida_dce_grid /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_dce_grid
cp /tmp/fernanda/commit_8d71876/$input_file/output_dce_grid.txt /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_dce_grid.txt
cp /tmp/fernanda/commit_8d71876/$input_file/diversity_grid_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/diversity_grid_fold_*
echo "5 >>>>>>>>> random_search.py"
time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/random_search.py -i $input_file -o saida_rs -e $num_classifiers -s $num_iterations &> output_rs.txt
cp /tmp/fernanda/commit_8d71876/$input_file/saida_rs /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_rs
cp /tmp/fernanda/commit_8d71876/$input_file/output_rs.txt /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_rs.txt
cp /tmp/fernanda/commit_8d71876/$input_file/rand_search_results_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
rm -rf /tmp/fernanda/commit_8d71876/$input_file/rand_search_results_fold_*
iteration=5
list='2 4 8 16 32'
for num_cores in $list;do 
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o saida_pdce_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_pdce_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_pdce_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_pdce_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_pdce_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_search_exec.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_brute_force_search_exec.py -i $input_file -o saida_pbf_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbf_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_pbf_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_pbf_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_pbf_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_pbf_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/pbfec_seq_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/pbfec_seq_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_brute_force_random_search_exec.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_brute_force_random_search_exec.py -i $input_file -o saida_pbfr_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pbfr_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_pbfr_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_pbfr_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_pbfr_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_pbfr_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/pbfec_rand_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/pbfec_rand_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o saida_pdce_grid_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores &> output_pdce_grid_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_pdce_grid_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_pdce_grid_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_pdce_grid_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_pdce_grid_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_grid_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_grid_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_0_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 0 &> output_prs_0_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_prs_0_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_prs_0_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_prs_0_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_prs_0_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_rs_results_ptype_0_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_rs_results_ptype_0_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_random_search.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_random_search.py -i $input_file -o saida_prs_1_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -p 1 &> output_prs_1_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/saida_prs_1_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/saida_prs_1_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/output_prs_1_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/output_prs_1_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_rs_results_ptype_1_fold_* /home/covoes/fernanda/commit_8d71876/$input_file
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_rs_results_ptype_1_fold_*
done
mkdir /home/covoes/fernanda/commit_8d71876/$input_file/v2
#list='16 32'
list='16'
for num_cores in $list;do 
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_ensemble.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_diversity_ensemble.py -i $input_file -o v2_saida_pdce_$num_cores -e $num_cores -s $num_iterations -c $num_cores &> v2_output_pdce_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/v2_saida_pdce_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/v2_saida_pdce_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/v2_output_pdce_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/v2_output_pdce_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_fold_* /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_fold_*
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> parallel_diversity_grid.py with $num_cores cores"
    time python3 -u /home/covoes/fernanda/commit_8d71876/evolutionary_ensemble/exec/parallel_diversity_grid.py -i $input_file -o v2_saida_pdce_grid_$num_cores -e $num_cores -s $num_iterations -c $num_cores &> v2_output_pdce_grid_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/v2_saida_pdce_grid_$num_cores /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/v2_saida_pdce_grid_$num_cores
    cp /tmp/fernanda/commit_8d71876/$input_file/v2_output_pdce_grid_$num_cores.txt /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/v2_output_pdce_grid_$num_cores.txt
    cp /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_grid_fold_* /home/covoes/fernanda/commit_8d71876/$input_file/v2
    rm -rf /tmp/fernanda/commit_8d71876/$input_file/parallel_diversity_grid_fold_*
done
popd
deactivate
popd