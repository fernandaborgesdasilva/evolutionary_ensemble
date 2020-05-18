import pandas as pd
import sys
from glob import glob

def get_brute_force_time(file, df):
    for iteration in range(0,10):
        file_name = file + str(iteration) + '*.csv'
        for f in glob(file_name):
            print(">>>>> file_name >>>>>",f)
            df_ini = pd.read_csv(f)
            df_ini["start_time"] = pd.to_datetime(df_ini["start_time"])
            df_ini["end_time"] = pd.to_datetime(df_ini["end_time"])
            start = min(df_ini['start_time'])
            end = max(df_ini['start_time'])
            total_time = pd.Timedelta(end - start).seconds
            df = df.append({'file_name': f,
                            'time_sec': 1,
                            'best_ensemble_accuracy': df_ini.head(1)["best_ensemble_accuracy"].item()}, ignore_index=True)
            max_fitness_aux = df_ini.head(1)["best_ensemble_accuracy"].item()
            first_end = round(df_ini.head(1)["total_time_ms"].item()/1000)+1
            for i in range(first_end,total_time + 1):
                up_to_time = start + pd.Timedelta(seconds=i)
                df_aux = df_ini[df_ini["end_time"] <= up_to_time]
                max_fitness = max(df_aux["best_ensemble_accuracy"])
                if (max_fitness_aux < max_fitness):
                    df = df.append({'file_name': f,
                                    'time_sec': i, 
                                    'best_ensemble_accuracy': max_fitness}, ignore_index=True)
                    max_fitness_aux = max_fitness
            df = df.append({'file_name': f,
                            'time_sec': total_time,
                            'best_ensemble_accuracy': max_fitness_aux}, ignore_index=True)
    return df

def get_diversity_time(file, df):
    for iteration in range(0,10):
        file_name = file + str(iteration) + '*.csv'
        for f in glob(file_name):
            print("\n\n>>>>> file_name >>>>>",f)
            df_ini = pd.read_csv(f)
            df_ini["start_time"] = pd.to_datetime(df_ini["start_time"])
            df_ini["end_time"] = pd.to_datetime(df_ini["end_time"])
            start = min(df_ini['start_time'])
            end = max(df_ini['start_time'])
            total_time = pd.Timedelta(end - start).seconds
            df = df.append({'file_name': f,
                            'time_sec': 1,
                            'best_ensemble_accuracy': df_ini.head(1)["best_ensemble_accuracy"].item()}, ignore_index=True)
            last_fitness_aux = df_ini.head(1)["best_ensemble_accuracy"].item()
            first_end = round(df_ini.head(1)["total_time_ms"].item()/1000)+1
            for i in range(first_end,total_time + 1):
                up_to_time = start + pd.Timedelta(seconds=i)
                df_aux = df_ini[df_ini["end_time"] <= up_to_time]
                max_val_end_time = max(df_aux["end_time"])
                last_fitness = df_aux[pd.to_datetime(df_aux["end_time"]) == max_val_end_time]["best_ensemble_accuracy"].item()
                if (last_fitness_aux != last_fitness):
                    last_fitness_aux = last_fitness
                    df = df.append({'file_name': f,
                                    'time_sec': i,
                                    'best_ensemble_accuracy': last_fitness}, ignore_index=True)
            df = df.append({'file_name': f,
                            'time_sec': total_time,
                            'best_ensemble_accuracy': last_fitness_aux}, ignore_index=True)
    return df

def main(args):

    brute_force_files = ["breast/bfec_rand_results_iter_",
                         "breast/bfec_seq_results_iter_",
                         "breast/pbfec_rand_results_iter_",
                         "breast/pbfec_seq_results_iter_"
                        ]

    diversity_files = ["breast/diversity_results_iter_",
                       "breast/parallel_diversity_results_iter_"
                      ]

    df = pd.DataFrame(columns=['file_name', 'time_sec', 'best_ensemble_accuracy'])

    for f in brute_force_files:
        df = get_brute_force_time(f,df)

    for f in diversity_files:
        df = get_diversity_time(f,df)

    df.to_csv("results_analysis.csv", index = None, header=True)

if __name__ == "__main__":
    main(sys.argv[1:])
