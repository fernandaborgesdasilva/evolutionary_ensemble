import pandas as pd
import sys
from glob import glob

def get_time(file, df):
    for col in [0,1]:
        for mut in [0,1]:
            for fold in range(0,5):
                for iteration in range(0,10):
                    file_name = file + str(fold) + '_col_' + str(col) + '_mutation_' + str(mut) + '_iter_' + str(iteration) + '*.csv'
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
                                        'ensemble_accuracy': df_ini.head(1)["ensemble_accuracy"].item()}, ignore_index=True)
                        max_accuracy_aux = df_ini.head(1)["ensemble_accuracy"].item()
                        first_end = round(df_ini.head(1)["total_time_ms"].item()/1000)+1
                        if(first_end == 1):
                            first_end = 2
                        for i in range(first_end,total_time):
                            up_to_time = start + pd.Timedelta(seconds=i)
                            df_aux = df_ini[df_ini["end_time"] <= up_to_time]
                            max_accuracy = max(df_aux["ensemble_accuracy"])
                            if (max_accuracy_aux < max_accuracy):
                                df = df.append({'file_name': f,
                                                'time_sec': i, 
                                                'ensemble_accuracy': max_accuracy}, ignore_index=True)
                                max_accuracy_aux = max_accuracy
                        df = df.append({'file_name': f,
                                        'time_sec': total_time,
                                        'ensemble_accuracy': max_accuracy_aux}, ignore_index=True)
    return df

def main(args):

    files = ["captcha/pdce_fold_"]

    df = pd.DataFrame(columns=['file_name', 'time_sec', 'ensemble_accuracy'])

    for f in files:
        df = get_time(f,df)

    #for f in diversity_files:
    #    df = get_diversity_time(f,df)

    df.to_csv("results_analysis_captcha_random_search_2.csv", index = None, header=True)

if __name__ == "__main__":
    main(sys.argv[1:])