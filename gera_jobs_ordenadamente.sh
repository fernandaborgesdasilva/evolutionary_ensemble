MAIN_DIR="/home/covoes/fernanda/assync_test_v2"
bases=("breast" "iris" "${MAIN_DIR}/evolutionary_ensemble/exec/dados_captcha.csv")
stopTime=19448
nEstimator=10
exec_path="${MAIN_DIR}/evolutionary_ensemble/exec"
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
        source /home/covoes/env_v_p/bin/activate
        pushd $MAIN_DIR


        "

        header_paralelo="#!/bin/bash
#SBATCH -J pll-TP.$basef
#SBATCH -o log-pll-TP.%j.out
#SBATCH -e log-pll-TP.%j.err
#SBATCH -n 32
#SBATCH --mem=20GB
#SBATCH -w compute-0-9
        source /home/covoes/env_v_p/bin/activate
        pushd $MAIN_DIR
        "

        fim="
        deactivate
        popd"

        comandos_dec="
        pushd $basef
        time python3 -u $exec_path/diversity_ensemble.py -i $base -o saida_dec -e $nEstimator -s $stopTime
        popd
        "

        comandos_forca_bruta="
        pushd $basef
        time python3 -u $exec_path/brute_force_search_exec.py -i $base -o saida_forca_bruta -e $nEstimator -s $stopTime
        popd
        "

        comandos_forca_bruta_random="
        pushd $basef
        time python3 -u $exec_path/brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria -e $nEstimator -s $stopTime
        popd
        "

        comandos_forca_bruta_parallel_2_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_2_cores -e $nEstimator -s $stopTime -c 2
        popd
        "

        comandos_forca_bruta_random_parallel_2_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_2_cores -e $nEstimator -s $stopTime -c 2
        popd
        "

        comandos_dec_parallel_2_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_2_cores -e $nEstimator -s $stopTime -c 2
        popd
        "

        comandos_forca_bruta_parallel_4_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_4_cores -e $nEstimator -s $stopTime -c 4
        popd
        "

        comandos_forca_bruta_random_parallel_4_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_4_cores -e $nEstimator -s $stopTime -c 4
        popd
        "

        comandos_dec_parallel_4_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_4_cores -e $nEstimator -s $stopTime -c 4
        popd
        "

        comandos_forca_bruta_parallel_8_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_8_cores -e $nEstimator -s $stopTime -c 8
        popd
        "

        comandos_forca_bruta_random_parallel_8_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_8_cores -e $nEstimator -s $stopTime -c 8
        popd
        "

        comandos_dec_parallel_8_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_8_cores -e $nEstimator -s $stopTime -c 8
        popd
        "

        comandos_forca_bruta_parallel_16_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_16_cores -e $nEstimator -s $stopTime -c 16
        popd
        "

        comandos_forca_bruta_random_parallel_16_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_16_cores -e $nEstimator -s $stopTime -c 16
        popd
        "

        comandos_dec_parallel_16_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_16_cores -e $nEstimator -s $stopTime -c 16
        popd
        "

        comandos_forca_bruta_parallel_32_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_search_exec.py -i $base -o saida_forca_bruta_pl_32_cores -e $nEstimator -s $stopTime -c 32
        popd
        "

        comandos_forca_bruta_random_parallel_32_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_brute_force_random_search_exec.py -i $base -o saida_forca_bruta_aleatoria_pl_32_cores -e $nEstimator -s $stopTime -c 32
        popd
        "

        comandos_dec_parallel_32_cores="
        pushd $basef
        time python3 -u $exec_path/parallel_diversity_ensemble.py -i $base -o saida_dec_pl_32_cores -e $nEstimator -s $stopTime -c 32
        popd
        "

        echo "${header//TP/DEC}" > job_01_dec_$basef
        echo "$comandos_dec" >> job_01_dec_$basef
        echo "$fim" >> job_01_dec_$basef

        echo "${header//TP/FB}" > job_02_forca_bruta_$basef
        echo "$comandos_forca_bruta" >> job_02_forca_bruta_$basef
        echo "$fim" >> job_02_forca_bruta_$basef

        echo "${header//TP/FBR}" > job_03_forca_bruta_random_$basef
        echo "$comandos_forca_bruta_random" >> job_03_forca_bruta_random_$basef
        echo "$fim" >> job_03_forca_bruta_random_$basef

        echo "${header_paralelo//TP/DEC}" > job_04_dec_paralelo_2_cores_$basef
        echo "$comandos_dec_parallel_2_cores" >> job_04_dec_paralelo_2_cores_$basef
        echo "$fim" >> job_04_dec_paralelo_2_cores_$basef

        echo "${header_paralelo//TP/FB}" > job_05_forca_bruta_paralelo_2_cores_$basef
        echo "$comandos_forca_bruta_parallel_2_cores" >> job_05_forca_bruta_paralelo_2_cores_$basef
        echo "$fim" >> job_05_forca_bruta_paralelo_2_cores_$basef

        echo "${header_paralelo//TP/FBR}" > job_06_forca_bruta_random_paralelo_2_cores_$basef
        echo "$comandos_forca_bruta_random_parallel_2_cores" >> job_06_forca_bruta_random_paralelo_2_cores_$basef
        echo "$fim" >> job_06_forca_bruta_random_paralelo_2_cores_$basef

        echo "${header_paralelo//TP/DEC}" > job_07_dec_paralelo_4_cores_$basef
        echo "$comandos_dec_parallel_4_cores" >> job_07_dec_paralelo_4_cores_$basef
        echo "$fim" >> job_07_dec_paralelo_4_cores_$basef

        echo "${header_paralelo//TP/FB}" > job_08_forca_bruta_paralelo_4_cores_$basef
        echo "$comandos_forca_bruta_parallel_4_cores" >> job_08_forca_bruta_paralelo_4_cores_$basef
        echo "$fim" >> job_08_forca_bruta_paralelo_4_cores_$basef

        echo "${header_paralelo//TP/FBR}" > job_09_forca_bruta_random_paralelo_4_cores_$basef
        echo "$comandos_forca_bruta_random_parallel_4_cores" >> job_09_forca_bruta_random_paralelo_4_cores_$basef
        echo "$fim" >> job_09_forca_bruta_random_paralelo_4_cores_$basef

        echo "${header_paralelo//TP/DEC}" > job_10_dec_paralelo_8_cores_$basef
        echo "$comandos_dec_parallel_8_cores" >> job_10_dec_paralelo_8_cores_$basef
        echo "$fim" >> job_10_dec_paralelo_8_cores_$basef

        echo "${header_paralelo//TP/FB}" > job_11_forca_bruta_paralelo_8_cores_$basef
        echo "$comandos_forca_bruta_parallel_8_cores" >> job_11_forca_bruta_paralelo_8_cores_$basef
        echo "$fim" >> job_11_forca_bruta_paralelo_8_cores_$basef

        echo "${header_paralelo//TP/FBR}" > job_12_forca_bruta_random_paralelo_8_cores_$basef
        echo "$comandos_forca_bruta_random_parallel_8_cores" >> job_12_forca_bruta_random_paralelo_8_cores_$basef
        echo "$fim" >> job_12_forca_bruta_random_paralelo_8_cores_$basef

        echo "${header_paralelo//TP/DEC}" > job_13_dec_paralelo_16_cores_$basef
        echo "$comandos_dec_parallel_16_cores" >> job_13_dec_paralelo_16_cores_$basef
        echo "$fim" >> job_13_dec_paralelo_16_cores_$basef

        echo "${header_paralelo//TP/FB}" > job_14_forca_bruta_paralelo_16_cores_$basef
        echo "$comandos_forca_bruta_parallel_16_cores" >> job_14_forca_bruta_paralelo_16_cores_$basef
        echo "$fim" >> job_14_forca_bruta_paralelo_16_cores_$basef

        echo "${header_paralelo//TP/FBR}" > job_15_forca_bruta_random_paralelo_16_cores_$basef
        echo "$comandos_forca_bruta_random_parallel_16_cores" >> job_15_forca_bruta_random_paralelo_16_cores_$basef
        echo "$fim" >> job_15_forca_bruta_random_paralelo_16_cores_$basef

        echo "${header_paralelo//TP/DEC}" > job_16_dec_paralelo_32_cores_$basef
        echo "$comandos_dec_parallel_32_cores" >> job_16_dec_paralelo_32_cores_$basef
        echo "$fim" >> job_16_dec_paralelo_32_cores_$basef

        echo "${header_paralelo//TP/FB}" > job_17_forca_bruta_paralelo_32_cores_$basef
        echo "$comandos_forca_bruta_parallel_32_cores" >> job_17_forca_bruta_paralelo_32_cores_$basef
        echo "$fim" >> job_17_forca_bruta_paralelo_32_cores_$basef

        echo "${header_paralelo//TP/FBR}" > job_18_forca_bruta_random_paralelo_32_cores_$basef
        echo "$comandos_forca_bruta_random_parallel_32_cores" >> job_18_forca_bruta_random_paralelo_32_cores_$basef
        echo "$fim" >> job_18_forca_bruta_random_paralelo_32_cores_$basef

done
