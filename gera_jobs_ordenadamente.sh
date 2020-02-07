bases=("breast" "iris" "/home/covoes/fernanda/evolutionary_ensemble/exec/dados_captcha.csv")
stopTime=19448
nEstimator=10
exec_path="/home/covoes/fernanda/evolutionary_ensemble/exec"
for base in "${bases[@]}"   
do

	basef=`basename $base .csv`
	mkdir -p $basef
	header="#!/bin/bash
#SBATCH -J seq-TP.$basef
#SBATCH -o log-seq-TP.%j.out
#SBATCH -e log-seq-TP.%j.err
#SBATCH --mem=20GB
#SBATCH -n 32
#SBATCH -w compute-0-9
	source env_v_p/bin/activate
	pushd env_v_p


	"

	header_paralelo="#!/bin/bash
#SBATCH -J pll-TP.$basef
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

	comandos_dec="
	pushd $basef
	python3 -u $exec_path/diversity_ensemble.py -i $base -o saida_dec -e $nEstimator -s $stopTime
	popd
	"
	
	comandos_forca_bruta="
	pushd $basef
	python3 -u $exec_path/brute_force_search_exec.py -i $base -o saida_forca_bruta -e $nEstimator -s $stopTime
	popd
	"

	comandos_forca_bruta_random="
	pushd $basef
	python3 -u $exec_path/brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria -e $nEstimator -s $stopTime
	popd
	"
	
	comandos_forca_bruta_parallel_2_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_2_cores -e $nEstimator -s $stopTime -c 2
	popd
	"

	comandos_forca_bruta_random_parallel_2_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_2_cores -e $nEstimator -s $stopTime -c 2
	popd
	"
	
	comandos_dec_parallel_2_cores="
	pushd $basef
	python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_2_cores -e $nEstimator -s $stopTime -c 2
	popd
	"

	comandos_forca_bruta_parallel_4_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_4_cores -e $nEstimator -s $stopTime -c 4
	popd
	"

	comandos_forca_bruta_random_parallel_4_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_4_cores -e $nEstimator -s $stopTime -c 4
	popd
	"
	
	comandos_dec_parallel_4_cores="
	pushd $basef
	python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_4_cores -e $nEstimator -s $stopTime -c 4
	popd
	"

	comandos_forca_bruta_parallel_8_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_8_cores -e $nEstimator -s $stopTime -c 8
	popd
	"

	comandos_forca_bruta_random_parallel_8_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_8_cores -e $nEstimator -s $stopTime -c 8
	popd
	"
	
	comandos_dec_parallel_8_cores="
	pushd $basef
	python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_8_cores -e $nEstimator -s $stopTime -c 8
	popd
	"

	comandos_forca_bruta_parallel_16_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_16_cores -e $nEstimator -s $stopTime -c 16
	popd
	"

	comandos_forca_bruta_random_parallel_16_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_16_cores -e $nEstimator -s $stopTime -c 16
	popd
	"
	
	comandos_dec_parallel_16_cores="
	pushd $basef
	python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_16_cores -e $nEstimator -s $stopTime -c 16
	popd
	"

	comandos_forca_bruta_parallel_32_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_32_cores -e $nEstimator -s $stopTime -c 32
	popd
	"

	comandos_forca_bruta_random_parallel_32_cores="
	pushd $basef
	python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_32_cores -e $nEstimator -s $stopTime -c 32
	popd
	"
	
	comandos_dec_parallel_32_cores="
	pushd $basef
	python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_32_cores -e $nEstimator -s $stopTime -c 32
	popd
	"
	
	echo "${header//TP/DEC}" > job_1_dec_$basef
	echo "$comandos_dec" >> job_1_dec_$basef
	echo "$fim" >> job_1_dec_$basef

	echo "${header//TP/FB}" > job_2_forca_bruta_$basef
	echo "$comandos_forca_bruta" >> job_2_forca_bruta_$basef
	echo "$fim" >> job_2_forca_bruta_$basef

	echo "${header//TP/FBR}" > job_3_forca_bruta_random_$basef
	echo "$comandos_forca_bruta_random" >> job_3_forca_bruta_random_$basef
	echo "$fim" >> job_3_forca_bruta_random_$basef

	echo "${header//TP/FBR}" > job_4_dec_paralelo_2_cores_$basef
	echo "$comandos_dec_parallel_2_cores" >> job_4_dec_paralelo_2_cores_$basef
	echo "$fim" >> job_4_dec_paralelo_2_cores_$basef

	echo "${header//TP/FBR}" > job_5_forca_bruta_paralelo_2_cores_$basef
	echo "$comandos_forca_bruta_parallel_2_cores" >> job_5_forca_bruta_paralelo_2_cores_$basef
	echo "$fim" >> job_5_forca_bruta_paralelo_2_cores_$basef

	echo "${header//TP/FBR}" > job_6_forca_bruta_random_paralelo_2_cores_$basef
	echo "$comandos_forca_bruta_random_parallel_2_cores" >> job_6_forca_bruta_random_paralelo_2_cores_$basef
	echo "$fim" >> job_6_forca_bruta_random_paralelo_2_cores_$basef

	echo "${header//TP/FBR}" > job_7_dec_paralelo_4_cores_$basef
	echo "$comandos_dec_parallel_4_cores" >> job_7_dec_paralelo_4_cores_$basef
	echo "$fim" >> job_7_dec_paralelo_4_cores_$basef

	echo "${header//TP/FBR}" > job_8_forca_bruta_paralelo_4_cores_$basef
	echo "$comandos_forca_bruta_parallel_4_cores" >> job_8_forca_bruta_paralelo_4_cores_$basef
	echo "$fim" >> job_8_forca_bruta_paralelo_4_cores_$basef

	echo "${header//TP/FBR}" > job_9_forca_bruta_random_paralelo_4_cores_$basef
	echo "$comandos_forca_bruta_random_parallel_4_cores" >> job_9_forca_bruta_random_paralelo_4_cores_$basef
	echo "$fim" >> job_9_forca_bruta_random_paralelo_4_cores_$basef

	echo "${header//TP/FBR}" > job_10_dec_paralelo_8_cores_$basef
	echo "$comandos_dec_parallel_8_cores" >> job_10_dec_paralelo_8_cores_$basef
	echo "$fim" >> job_10_dec_paralelo_8_cores_$basef

	echo "${header//TP/FBR}" > job_11_forca_bruta_paralelo_8_cores_$basef
	echo "$comandos_forca_bruta_parallel_8_cores" >> job_11_forca_bruta_paralelo_8_cores_$basef
	echo "$fim" >> job_11_forca_bruta_paralelo_8_cores_$basef

	echo "${header//TP/FBR}" > job_12_forca_bruta_random_paralelo_8_cores_$basef
	echo "$comandos_forca_bruta_random_parallel_8_cores" >> job_12_forca_bruta_random_paralelo_8_cores_$basef
	echo "$fim" >> job_12_forca_bruta_random_paralelo_8_cores_$basef

	echo "${header//TP/FBR}" > job_13_dec_paralelo_16_cores_$basef
	echo "$comandos_dec_parallel_16_cores" >> job_13_dec_paralelo_16_cores_$basef
	echo "$fim" >> job_13_dec_paralelo_16_cores_$basef

	echo "${header//TP/FBR}" > job_14_forca_bruta_paralelo_16_cores_$basef
	echo "$comandos_forca_bruta_parallel_16_cores" >> job_14_forca_bruta_paralelo_16_cores_$basef
	echo "$fim" >> job_14_forca_bruta_paralelo_16_cores_$basef

	echo "${header//TP/FBR}" > job_15_forca_bruta_random_paralelo_16_cores_$basef
	echo "$comandos_forca_bruta_random_parallel_16_cores" >> job_15_forca_bruta_random_paralelo_16_cores_$basef
	echo "$fim" >> job_15_forca_bruta_random_paralelo_16_cores_$basef

	echo "${header//TP/FBR}" > job_16_dec_paralelo_32_cores_$basef
	echo "$comandos_dec_parallel_32_cores" >> job_16_dec_paralelo_32_cores_$basef
	echo "$fim" >> job_16_dec_paralelo_32_cores_$basef

	echo "${header//TP/FBR}" > job_17_forca_bruta_paralelo_32_cores_$basef
	echo "$comandos_forca_bruta_parallel_32_cores" >> job_17_forca_bruta_paralelo_32_cores_$basef
	echo "$fim" >> job_17_forca_bruta_paralelo_32_cores_$basef

	echo "${header//TP/FBR}" > job_18_forca_bruta_random_paralelo_32_cores_$basef
	echo "$comandos_forca_bruta_random_parallel_32_cores" >> job_18_forca_bruta_random_paralelo_32_cores_$basef
	echo "$fim" >> job_18_forca_bruta_random_paralelo_32_cores_$basef

done
