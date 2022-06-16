#!/bin/bash
#SBATCH -J captcha_island
#SBATCH -o captcha_island.%j.out
#SBATCH -e captcha_island.%j.err
#SBATCH -n 32
#SBATCH --mem=20GB
#SBATCH -w compute-0-5
export OPENBLAS_NUM_THREADS=1
input_file=captcha
captcha_dir=/home/covoes/fernanda/commit_77048b6/evolutionary_ensemble/exec/dados_captcha.csv
num_classifiers=10
num_iterations=19448
source /home/covoes/env_v_p/bin/activate
mkdir /tmp/fernanda/commit_77048b6/island/$input_file
mkdir /home/covoes/fernanda/commit_77048b6/island/$input_file
pushd /tmp/fernanda/commit_77048b6/island
iteration=0
lista_qtd_ilhas='2 4'
for qtd_ilhas in $lista_qtd_ilhas;do
    mkdir /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas
    mkdir /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas
    pushd /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas
        #lista_mig_interval='10 15 20'
        lista_mig_interval='10'
        for mig_interval in $lista_mig_interval;do
            mkdir /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval
            mkdir /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval
            pushd /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval
                #lista_mig_size='1 2 3 4 5'
                lista_mig_size='2'
                for mig_size in $lista_mig_size;do
                    mkdir /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size
                    mkdir /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size
                    pushd /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size
                    let iteration=$iteration+1
                    echo "$iteration >>>>>>>>> dce_island_model.py with $qtd_ilhas cores and with selection criteria iqual to 0"
                    echo "mig_interval = $mig_interval"
                    echo "mig_size = $mig_size"
                    time python3 -u /home/covoes/fernanda/commit_77048b6/evolutionary_ensemble/exec/dce_island_model.py -i $captcha_dir -o outputfile_dce_island_model_t0_$qtd_ilhas -e $num_classifiers -g $num_iterations -n$qtd_ilhas -m $mig_interval -s $mig_size -c $qtd_ilhas -t 0 &> stdout_dce_island_model_t0_$qtd_ilhas.txt
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/outputfile_dce_island_model_t0_$qtd_ilhas /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/outputfile_dce_island_model_t0_$qtd_ilhas
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/stdout_dce_island_model_t0_$qtd_ilhas.txt /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/stdout_dce_island_model_t0_$qtd_ilhas.txt
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/dce_island_sel_* /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/dce_island_sel_*
                    echo ""
                    let iteration=$iteration+1
                    echo "$iteration >>>>>>>>> dce_island_model.py with $qtd_ilhas cores and with selection criteria iqual to 1"
                    echo "mig_interval = $mig_interval"
                    echo "mig_size = $mig_size"
                    time python3 -u /home/covoes/fernanda/commit_77048b6/evolutionary_ensemble/exec/dce_island_model.py -i $captcha_dir -o outputfile_dce_island_model_t1_$qtd_ilhas -e $num_classifiers -g $num_iterations -n$qtd_ilhas -m $mig_interval -s $mig_size -c $qtd_ilhas -t 1 &> stdout_dce_island_model_t1_$qtd_ilhas.txt
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/outputfile_dce_island_model_t1_$qtd_ilhas /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/outputfile_dce_island_model_t1_$qtd_ilhas
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/stdout_dce_island_model_t1_$qtd_ilhas.txt /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/stdout_dce_island_model_t1_$qtd_ilhas.txt
                    cp /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/dce_island_sel_* /home/covoes/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/
                    rm -rf /tmp/fernanda/commit_77048b6/island/$input_file/qtd_ilhas_$qtd_ilhas/mig_interval_$mig_interval/mig_size_$mig_size/dce_island_sel_*
                    echo ""
                    popd
                done
            popd
        done
    popd
done
deactivate
popd