import pandas as pd
import sys
from glob import glob

def get_time(file, df, num_island):
    for sel in [0,1]:
        for fold in range(0,5):
            for iteration in range(0,10):
                for island in range(0, num_island):
                    file_name = file + str(sel) + '_fold_' + str(fold) + '_iter_' + str(iteration) + '_island_' + str(island) + '*.csv'
                    all_files = glob(file_name)
                    if len(all_files) > 0:
                        print("\n\n>>>>> file_name >>>>>",all_files[0])
                        df_ini = pd.read_csv(all_files[0])
                        if len(all_files) > 1:
                            for f in range(1, len(all_files)):
                                print(">>>>> file_name >>>>>",all_files[f])
                                df_ini = df_ini.append(pd.read_csv(all_files[f]))
                        df_ini["start_time"] = pd.to_datetime(df_ini["start_time"])
                        df_ini["end_time"] = pd.to_datetime(df_ini["end_time"])
                        start = min(df_ini['start_time'])
                        end = max(df_ini['start_time'])
                        total_time = pd.Timedelta(end - start).seconds
                        print("total_time = ", total_time)
                        df_ini_start = df_ini[df_ini['start_time'] == start]
                        df = df.append({'file_name': file_name,
                                        'time_sec': 1,
                                        'ensemble_accuracy': df_ini_start.head(1)["ensemble_accuracy"].item()}, ignore_index=True)
                        max_accuracy_aux = df_ini_start.head(1)["ensemble_accuracy"].item()
                        first_end = round(df_ini_start.head(1)["total_time_ms"].item()/1000)+1
                        if(first_end == 1):
                            first_end = 2
                        for i in range(first_end,total_time):
                            up_to_time = start + pd.Timedelta(seconds=i)
                            df_aux = df_ini[df_ini["end_time"] <= up_to_time]
                            max_accuracy = max(df_aux["ensemble_accuracy"])
                            if (max_accuracy_aux < max_accuracy):
                                df = df.append({'file_name': file_name,
                                                'time_sec': i, 
                                                'ensemble_accuracy': max_accuracy}, ignore_index=True)
                                max_accuracy_aux = max_accuracy
                        df = df.append({'file_name': file_name,
                                        'time_sec': total_time,
                                        'ensemble_accuracy': max_accuracy_aux}, ignore_index=True)
    return df

def main(args):

    files = ["island/breast/qtd_ilhas_2/mig_interval_10/mig_size_2/dce_island_sel_",
             "island/breast/qtd_ilhas_4/mig_interval_10/mig_size_2/dce_island_sel_"
            ]

    df = pd.DataFrame(columns=['file_name', 'time_sec', 'ensemble_accuracy'])

    for f, i in zip(files, [2,4]):
        df = get_time(f, df, i)

    #for f in diversity_files:
    #    df = get_diversity_time(f,df)

    df.to_csv("results_analysis_islands_breast.csv", index = None, header=True)

if __name__ == "__main__":
    main(sys.argv[1:])