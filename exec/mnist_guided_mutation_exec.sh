#!/bin/bash
#SBATCH -J mnist_guided
#SBATCH -o mnist_guided.%j.out
#SBATCH -e mnist_guided.%j.err
#SBATCH -n 32
#SBATCH --mem=20GB
#SBATCH -w compute-0-13
export OPENBLAS_NUM_THREADS=1
input_file=mnist
mnist_dir=/home/covoes/fernanda/commit_d9c366c/mnist_amos_3_8_9.csv
num_classifiers=10
num_iterations=19448
source /home/covoes/env_v_p/bin/activate
#mkdir /home/covoes/fernanda/commit_d9c366c/$input_file
#rm -rf /tmp/fernanda/commit_d9c366c/$input_file
#mkdir /tmp/fernanda/commit_d9c366c/$input_file
pushd /tmp/fernanda/commit_d9c366c/
pushd $input_file
iteration=0
list='1 4 8 2 16 32'
for num_cores in $list;do 
    #let iteration=$iteration+1
    #echo "$iteration >>>>>>>>> pdce_mnist.py with $num_cores cores"
    #echo "The features will be considered as genes by the algorithm"
    #echo "The mutation will be guided by the fitness"
    #time python3 -u /home/covoes/fernanda/commit_d9c366c/evolutionary_ensemble/exec/pdce_mnist.py -i $mnist_dir -o output_pdce_g1_m1_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -g 1 -m 1 &> stdout_pdce_g1_m1_$num_cores.txt
    #cp /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g1_m1_$num_cores /home/covoes/fernanda/commit_d9c366c/$input_file
    #rm -rf /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g1_m1_$num_cores
    #cp /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g1_m1_$num_cores.txt /home/covoes/fernanda/commit_d9c366c/$input_file
    #rm -rf /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g1_m1_$num_cores.txt
    #cp /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_* /home/covoes/fernanda/commit_d9c366c/$input_file
    #rm -rf /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_*
    echo ""
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> pdce_mnist.py with $num_cores cores"
    echo "The features will not be considered as genes by the algorithm"
    echo "The mutation will be guided by the fitness"
    time python3 -u /home/covoes/fernanda/commit_d9c366c/evolutionary_ensemble/exec/pdce_mnist.py -i $mnist_dir -o output_pdce_g0_m1_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -g 0 -m 1 &> stdout_pdce_g0_m1_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g0_m1_$num_cores /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g0_m1_$num_cores
    cp /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g0_m1_$num_cores.txt /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g0_m1_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_* /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_*
    echo ""
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> pdce_mnist.py with $num_cores cores"
    echo "The features will be considered as genes by the algorithm"
    echo "The mutation will not be guided by the fitness"
    time python3 -u /home/covoes/fernanda/commit_d9c366c/evolutionary_ensemble/exec/pdce_mnist.py -i $mnist_dir -o output_pdce_g1_m0_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -g 1 -m 0 &> stdout_pdce_g1_m0_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g1_m0_$num_cores /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g1_m0_$num_cores
    cp /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g1_m0_$num_cores.txt /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g1_m0_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_* /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_*
    echo ""
    let iteration=$iteration+1
    echo "$iteration >>>>>>>>> pdce_mnist.py with $num_cores cores"
    echo "The features will not be considered as genes by the algorithm"
    echo "The mutation will not be guided by the fitness"
    time python3 -u /home/covoes/fernanda/commit_d9c366c/evolutionary_ensemble/exec/pdce_mnist.py -i $mnist_dir -o output_pdce_g0_m0_$num_cores -e $num_classifiers -s $num_iterations -c $num_cores -g 0 -m 0 &> stdout_pdce_g0_m0_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g0_m0_$num_cores /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/output_pdce_g0_m0_$num_cores
    cp /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g0_m0_$num_cores.txt /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/stdout_pdce_g0_m0_$num_cores.txt
    cp /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_* /home/covoes/fernanda/commit_d9c366c/$input_file
    rm -rf /tmp/fernanda/commit_d9c366c/$input_file/pdce_fold_*
    echo ""
done
popd
deactivate
popd