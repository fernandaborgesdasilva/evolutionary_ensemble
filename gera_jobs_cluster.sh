bases=("breast" "iris" "/home/covoes/evolutionary_ensemble/exec/dados_captcha.csv")
stopTime=1000
nEstimator=10
exec_path="/home/covoes/env_v_p/evolutionary_ensemble/exec"
for base in "${bases[@]}"   
do

	basef=`basename $base .csv`
	mkdir -p $basef
	header="#!/bin/bash
#SBATCH -J seq-$basef
#SBATCH -o log-seq-TP.%j.out
#SBATCH -e log-seq-TP.%j.err
#SBATCH --mem=20GB
#SBATCH -n 32
#SBATCH -w compute-0-9
	source env_v_p/bin/activate
	pushd env_v_p


	"

	header_paralelo="#!/bin/bash
#SBATCH -J pll-$basef
#SBATCH -o log-pll-TP.%j.out
#SBATCH -e log-pll-TP.%j.err
#SBATCH -n 32
#SBATCH --mem=20GB
#SBATCH -w compute-0-9
	source env_v_p/bin/activate
	pushd env_v_p
	"

	fim="
	deactivate
	popd"



	comandos_forca_bruta_random="

	pushd $basef
	python $exec_path/brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria -e $nEstimator -s $stopTime
	popd
	"
	comandos_forca_bruta_random_parallel="
	cores=(2 4 8 16 32)
	pushd $basef
	for i in \"\${cores[@]}\"
	do
		python $exec_patch/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl\$i -e $nEstimator -s $stopTime -c \$i
	done
	popd
	"

	comandos_forca_bruta="
	pushd $basef
	python $exec_path/brute_force_search_exec.py -i $base -o saida_forca_bruta -e $nEstimator -s $stopTime
	popd
	"
	comandos_forca_bruta_parallel="
	cores=(2 4 8 16 32)
	pushd $basef
	for i in \"\${cores[@]}\"
	do
		python $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl\$i -e $nEstimator -s $stopTime -c \$i
	done
	popd
	"


	comandos_dec="
	pushd $basef
	python $exec_path/diversity_ensemble.py -i $base -o saida_dec -e $nEstimator -s $stopTime
	popd
	"
	comandos_dec_parallel="
	cores=(2 4 8 16 32)
	pushd $basef
	for i in \"\${cores[@]}\"
	do
		python $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl\$i -e $nEstimator -s $stopTime -c \$i
	done
	popd
	"

	echo "${header//TP/FB}" > job_forca_bruta_$basef
	echo "$comandos_forca_bruta" >> job_forca_bruta_$basef
	echo "$fim" >> job_forca_bruta_$basef

	echo "${header//TP/FBR}" > job_forca_bruta_random_$basef
	echo "$comandos_forca_bruta_random" >> job_forca_bruta_random_$basef
	echo "$fim" >> job_forca_bruta_random_$basef

	echo "${header_paralelo//TP/FB}" > job_forca_bruta_paralelo_$basef
	echo "$comandos_forca_bruta_parallel" >> job_forca_bruta_paralelo_$basef
	echo "$fim" >> job_forca_bruta_paralelo_$basef

	echo "${header_paralelo//TP/FBR}" > job_forca_bruta_random_paralelo_$basef
	echo "$comandos_forca_bruta_random_parallel" >> job_forca_bruta_random_paralelo_$basef
	echo "$fim" >> job_forca_bruta_random_paralelo_$basef

	echo "${header//TP/DEC}" > job_dec_$basef
	echo "$comandos_dec" >> job_dec_$basef
	echo "$fim" >> job_dec_$basef

	echo "${header_paralelo//TP/DEC}" > job_dec_paralelo_$basef
	echo "$comandos_dec_parallel" >> job_dec_paralelo_$basef
	echo "$fim" >> job_dec_paralelo_$basef

done
