import json
import os
import csv
import re
import random
from datetime import datetime, timedelta
from tqdm import tqdm


MODE = "train"
DATA = "LifeSnaps" # "PMData", "GLOBEM", "AW_FB"


def avg(_list):
    return sum(_list) / len(_list)

def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def csv_reader(file_name):
    f = open(file_name, 'r')
    reader = csv.reader(f)
    # f.close()
    return reader

def convert_to_lowercase(input_string):
    return input_string.lower()

def has_alphabets(input_string):
    return any(char.isalpha() for char in input_string)

def has_numbers(input_string):
    has_numbers = any(char.isdigit() for char in input_string)
    return has_numbers

def extract_words_inside_brackets(input_string):
    pattern = r"\[\'([^\']+)\'\]|\[\"([^\"]+)\"\]"

    matches = re.findall(pattern, input_string)

    extracted_words = ' '.join([word for match in matches for word in match if word])

    return extracted_words


if MODE == "train":
    if DATA == "LifeSnaps":
        print("[INFO] Dataset:", DATA)
        SUBTASK = "stress_resilience" #"sleep_disorder"
        print("[INFO] Subtask:", SUBTASK)
        DATA_PATH = "data/life_snaps"

        if SUBTASK == "stress_resilience":
            final_data = []

            # 1) read panas data
            with open('{}/scored_surveys/panas.csv'.format(DATA_PATH), newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

                stress_resilience_list = []
                for idx,row in enumerate(csv_reader):
                    if idx==0:
                        continue
                    # print("row:", row)
                    tmp = row[0].split(",")
                    user_id = tmp[1]
                    date = datetime.strptime(tmp[3], '%Y-%m-%d')
                    positive_affect_score = int(tmp[4]) # 10 ~ 50
                    negative_affect_score = int(tmp[5]) # 10 ~ 50

                    stress_resilience = positive_affect_score / negative_affect_score # 0.2 ~ 5
                    stress_resilience_list.append({'user_id':user_id, 'date': date, 'stress_resilience': stress_resilience, 'positive_affect_score': positive_affect_score, 'negative_affect_score': negative_affect_score})

            # 2) read fitbit data
            with open('{}/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv'.format(DATA_PATH), newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                fitbit_list = []
                for idx,row in enumerate(csv_reader):
                    if idx == 0:
                        continue
                    
                    tmp = row[0].split(",")
                    try:
                        user_id = tmp[1]
                        date = datetime.strptime(tmp[2], '%Y-%m-%d')
                        activity_type = tmp[17]
                        lightly_active_minutes = float(tmp[19])
                        moderately_active_minutes = float(tmp[20])
                        very_active_minutes = float(tmp[21])
                        sleep_duration = float(tmp[26])
                        sleep_efficiency = float(tmp[31])
                        sleep_deep_ratio = float(tmp[32])
                        sleep_light_ratio = float(tmp[34])
                        sleep_rem_ratio = float(tmp[35])
                        stress_score = float(tmp[8])
                    except Exception as e:
                        # print(e)
                        continue

                    # print("stress_score:", stress_score)

                    fitbit_list.append({'user_id': user_id, 'date': date, 'activity_type':activity_type, 'lightly_active_minutes': lightly_active_minutes, 'moderately_active_minutes': moderately_active_minutes, 'very_active_minutes': very_active_minutes, 'sleep_duration': sleep_duration, 'sleep_efficiency': sleep_efficiency, 'sleep_deep_ratio':sleep_deep_ratio, 'sleep_light_ratio': sleep_light_ratio, 'sleep_rem_ratio':sleep_rem_ratio, 'stress_score':stress_score})

            # iterate through the label (stress resilience)
            for item1 in stress_resilience_list:
                user1 = item1['user_id']
                stress_resilience_index = item1['stress_resilience'] # what we want to predict
                tmp_positive_affect_score = item1['positive_affect_score']
                tmp_negative_affect_score = item1['negative_affect_score']

                tmp_activity_type = []
                tmp_lightly_active_minutes = []
                tmp_moderately_active_minutes = []
                tmp_very_active_minutes = []
                tmp_sleep_duration = []
                tmp_sleep_efficiency = []
                tmp_sleep_deep_ratio = []
                tmp_sleep_light_ratio = []
                tmp_sleep_rem_ratio = []
                tmp_stress_score = []

                for item2 in fitbit_list:
                    user2 = item2['user_id']

                    if user1 == user2:
                        date1 = item1['date']
                        date2 = item2['date']
                        if (date1 > date2) and (date1 - date2) < timedelta(days=7):
                            # TODO: 1-week summary
                            tmp_activity_type.append(item2['activity_type'])
                            tmp_lightly_active_minutes.append(item2['lightly_active_minutes'])
                            tmp_moderately_active_minutes.append(item2['moderately_active_minutes'])
                            tmp_very_active_minutes.append(item2['very_active_minutes'])
                            tmp_sleep_duration.append(item2['sleep_duration'])
                            tmp_sleep_efficiency.append(item2['sleep_efficiency'])
                            tmp_sleep_deep_ratio.append(item2['sleep_deep_ratio'])
                            tmp_sleep_light_ratio.append(item2['sleep_light_ratio'])
                            tmp_sleep_rem_ratio.append(item2['sleep_rem_ratio'])
                            tmp_stress_score.append(item2['stress_score'])
                
                
                if len(tmp_activity_type) == 0:
                    tmp_activity_type = ["N/A"]
                if len(tmp_lightly_active_minutes) == 0:
                    tmp_lightly_active_minutes = [-1]
                if len(tmp_moderately_active_minutes) == 0:
                    tmp_moderately_active_minutes = [-1]
                if len(tmp_very_active_minutes) == 0:
                    tmp_very_active_minutes = [-1]
                if len(tmp_sleep_duration) == 0:
                    tmp_sleep_duration = [-1]
                if len(tmp_sleep_efficiency) == 0:
                    tmp_sleep_efficiency = [-1]
                if len(tmp_sleep_deep_ratio) == 0:
                    tmp_sleep_deep_ratio = [-1]
                if len(tmp_sleep_light_ratio) == 0:
                    tmp_sleep_light_ratio = [-1]
                if len(tmp_sleep_rem_ratio) == 0:
                    tmp_sleep_rem_ratio = [-1]
                if len(tmp_stress_score) == 0:
                    tmp_stress_score = [-1]
                
                I = "Answer this question truthfully"
                Q = "Given the following 7-days averaged data, predict the Stress Resilience Index (SRI) between 0.2 and 5. [Stress Score]: {} out of 100, [Positive Affect Score]: {} out of 50, [Negative Affect Score]: {} out of 50, [Lightly Active Minutes]: {}, [Moderately Active Minutes]: {}, [Very Active Minutes]: {}, [Sleep Efficiency]: {}, [Sleep Deep Ratio]: {}, [Sleep Light Ratio]: {}, [Sleep REM Ratio]: {}.".format(tmp_stress_score, tmp_positive_affect_score, tmp_negative_affect_score, tmp_lightly_active_minutes, tmp_moderately_active_minutes, tmp_very_active_minutes, tmp_sleep_efficiency, tmp_sleep_deep_ratio,tmp_sleep_light_ratio, tmp_sleep_rem_ratio)
                A = "{:.2f}".format(stress_resilience_index)

                # NEW
                # I = "You are a personalized healthcare agent trained to predict {} which ranges from 0.2 to 5 based on physiological data and user information.".format(SUBTASK)
                # Q = "The recent 7-days sensor readings show: [Stress Score]: {} out of 100, [Positive Affect Score]: {} out of 50, [Negative Affect Score]: {} out of 50, [Lightly Active Minutes]: {} minutes, [Moderately Active Minutes]: {} minutes, [Very Active Minutes]: {} minutes, [Sleep Efficiency]: {}, [Sleep Deep Ratio]: {}, [Sleep Light Ratio]: {}, [Sleep REM Ratio]: {}; What would be the predicted stress resilience index?".format(str(tmp_stress_score), str(tmp_positive_affect_score), str(tmp_negative_affect_score), str(tmp_lightly_active_minutes), str(tmp_moderately_active_minutes), str(tmp_very_active_minutes), str(tmp_sleep_efficiency), str(tmp_sleep_deep_ratio), str(tmp_sleep_light_ratio), str(tmp_sleep_rem_ratio))
                # A = "The predicted stress resilience index is {:.2f}.".format(stress_resilience_index)

                final_data.append({'instruction':I, 'input':Q, 'output':A})        


        elif SUBTASK == "sleep_disorder":
            final_data = []
            
            # read fitbit data
            with open('{}/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv'.format(DATA_PATH), newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                fitbit_list = []
                for idx,row in enumerate(csv_reader):
                    if idx == 0:
                        continue
                    
                    tmp = row[0].split(",")
                    try:
                        user_id = tmp[1]
                        date = datetime.strptime(tmp[2], '%Y-%m-%d')
             
                        sleep_duration = float(tmp[26]) / 3600 / 1000
                        minutes_awake = float(tmp[29])
                        sleep_efficiency = float(tmp[31])
                        sleep_deep_ratio = float(tmp[32])
                        sleep_wake_ratio = float(tmp[33])
                        sleep_light_ratio = float(tmp[34])
                        sleep_rem_ratio = float(tmp[35])

                        rmssd = float(tmp[5])
                        spo2 = float(tmp[6])
                        full_sleep_breathing_rate = float(tmp[7])
                        bpm = float(tmp[18])
                        resting_hr = float(tmp[25])
                    
                    except Exception as e:
                        # print(e)
                        continue

                    # DSM-IV ICSD
                    print("sleep_efficiency:", sleep_efficiency)
                    print("minutes_awake:", minutes_awake)
                    
                    if sleep_efficiency < 85 and minutes_awake > 30:
                        sleep_disorder = 1
                    else:
                        sleep_disorder = 0
                    
                    print("sleep_disorder:", sleep_disorder)
                    print()

                    I = "Answer this question truthfully"
                    Q = "Given the following data, predict whether there exists sleep disorder (1) or not (0). [Sleep Duration]: {:.2f} hours, [Minutes Awake]: {} minutes, [Sleep Efficiency]: {}, [Sleep Deep Ratio]: {:.2f}, [Sleep Wake Ratio]: {:.2f}, [Sleep Light Ratio]: {:.2f}, [Sleep REM Ratio]: {:.2f}, [RMSSD]: {:.2f}, [SPO2]: {} %, [Full Sleep Breathing Rate]: {}, [BPM]: {:.2f}, [Resting Hour]: {:.2f} hours".format(sleep_duration, minutes_awake, sleep_efficiency, sleep_deep_ratio, sleep_wake_ratio, sleep_light_ratio, sleep_rem_ratio, rmssd, spo2, full_sleep_breathing_rate, bpm, resting_hr)
                    A = "{}".format(sleep_disorder)

                    # NEW
                    # I = "You are a personalized healthcare agent trained to predict the {} which is either 1 (exist) or 0 (does not exist) based on physiological data and user information.".format(SUBTASK)
                    # Q = "The recent sensor readings show: [Sleep Duration]: {:.2f} hours, [Minutes Awake]: {} minutes, [Sleep Efficiency]: {}, [Sleep Deep Ratio]: {:.2f}, [Sleep Wake Ratio]: {:.2f}, [Sleep Light Ratio]: {:.2f}, [Sleep REM Ratio]: {:.2f}, [RMSSD]: {:.2f}, [SPO2]: {} %, [Full Sleep Breathing Rate]: {}, [BPM]: {:.2f}, [Resting Hour]: {:.2f} hours; What would be the predicted sleep disorder?".format(sleep_duration, minutes_awake, sleep_efficiency, sleep_deep_ratio, sleep_wake_ratio, sleep_light_ratio, sleep_rem_ratio, rmssd, spo2, full_sleep_breathing_rate, bpm, resting_hr)
                    # A = "The predicted sleep disorder is {}.".format(sleep_disorder)

                    final_data.append({'instruction':I, 'input':Q, 'output':A})        


    elif DATA == "AW_FB":
        DATA_PATH = "medAlpaca/data/harvard_dataverse"
        SUBTASK = "activity"
        print("[INFO] Generating datasets for AW_FB (Harvard Dataverse) ...")
        print("[INFO] Subtask:", SUBTASK)

        with open('{}/aw_fb_data.csv'.format(DATA_PATH), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            final_data = []
            activities = []

            for idx,row in enumerate(csv_reader):
                if idx == 0:
                    continue

                try:
                    tmp = row[0].split(",")
                    age = int(tmp[2])
                    gender = int(tmp[3])
                    if gender == 1:
                        gender = "M"
                    elif gender == 0:
                        gender = "F"
                    height = tmp[4] + " cm"
                    weight = tmp[5] + " kg"
                    steps = float(tmp[6])
                    heart_rate = float(tmp[7])
                    calories = float(tmp[8])
                    distance = float(tmp[9])
                    entropy_heart = float(tmp[10])
                    entropy_steps = float(tmp[11])
                    resting_heart = float(tmp[12])
                    corr_heart_steps = float(tmp[13])
                    norm_heart = float(tmp[14])
                    intensity_karvonen = float(tmp[15]) 
                    sd_norm_heart = float(tmp[16])
                    steps_times_distance = float(tmp[17]) 
                    device = tmp[18]
                    
                    if len(row) == 3:
                        activity = row[0].split(",")[-1] + " " + row[1] + " " + row[2]
                    else:
                        activity = tmp[19] # label

                    activity = activity.replace('"', "")

                    if activity == "Self Pace walk":
                        met_value = 2
                    elif activity == "Sitting":
                        met_value = 1
                    elif activity == "Lying":
                        met_value = 0.5
                    elif activity == "Running 7 METs":
                        met_value = 7
                    elif activity == "Running 5 METs":
                        met_value = 5
                    elif activity == "Running 3 METs":
                        met_value = 3

                    duration = calories / (met_value * float(tmp[5]) / 200)

                    activities.append(activity)

                    # I = "Answer this question truthfully"

                    if SUBTASK == "activity":
                        # Q = "Predict the {} type among ['Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Sitting', 'Running 5 METs', 'Running 3 METs'] given the following information. [Age]: {}, [Gender]: {}, [Height]: {}, [Weight]: {}, [Steps]: {:.2f} steps, [Burned Calorories]: {:.2f} calories, [Heart Rate]: {:.2f} beats/min".format(SUBTASK, age, gender, height, weight, steps, calories, heart_rate)
                        # A = "{}".format(activity.replace('\"', ''))
                    
                        # NEW
                        I = "You are a personalized healthcare agent trained to predict the type of activity among ['Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Running 5 METs', 'Running 3 METs'] based on physiological data and user information."
                        Q = "The recent sensor readings show: [Steps]: {:.2f} steps, [Burned Calorories]: {:.2f} calories, [Heart Rate]: {:.2f} beats/min; What would be the predicted activity type?".format(steps, calories, heart_rate)
                        A = "The predicted activity type is {}".format(activity.replace('\"', ''))

                    elif SUBTASK == "calories":
                        # Q = "Predict the burned {} given the following information. [Age]: {}, [Gender]: {}, [Height]: {}, [Weight]: {}, [Steps]: {:.2f} steps, [Heart Rate]: {:.2f} beats/min".format(SUBTASK, age, gender, height, weight, steps, heart_rate)
                        # A = "{}".format(calories)

                        # NEW
                        I = "You are a personalized healthcare agent trained to predict the calorie burn based on physiological data and user information."
                        Q = "The recent sensor readings show: [Steps]: {:.2f} steps, [Heart Rate]: {:.2f} beats/min, [Duration]: {:.2f}, [Activity Type]: {}, [MET Value]: {}; What would be the predicted calorie burn?".format(steps, heart_rate, duration, activity, met_value)
                        A = "The predicted calorie burn is {}.".format(calories)
                    
                    final_data.append({'instruction':I, 'input':Q, 'output':A})
                
                except Exception as e:
                    print(e)
                    continue


    elif DATA == "PMData":
        print("[INFO] Generating datasets for PMData ...")
        """ 
            1) Readiness Prediction 
            2) Stress Estimation
            3) Calorie Consumption Prediction (food image) 
            4) Sleep Quality Prediction
        """
        participant_info = {
            'p1': [-1, -1, 'N/A', 'N/A', 'N/A']
        }
        
        DATA_PATH = "medAlpaca/data/pmdata"
        SUBTASK = "sleep_quality" # ['sleep_quality', 'stress', 'readiness', 'fatigue']:
        final_data = []
        print("[INFO] Subtask:", SUBTASK)
        for dir1 in tqdm(os.listdir(DATA_PATH)):
            if "." in dir1:
                continue
            tmp = participant_info[dir1]
            age = tmp[0]
            height = str(tmp[1]) + " cm"
            gender = tmp[2]
            fpath1 = os.path.join(DATA_PATH, dir1)

            if '.' not in dir1:      
                # print("[INFO] Participant:", dir1)      
                for dir2 in os.listdir(fpath1):
                    fpath2 = os.path.join(fpath1, dir2)

                    if dir2 == 'fitbit':
                        # [1] fitbit
                        # 1-1) calories.json
                        # 1-2) distance.json
                        
                        # 1-3) exercise.json
                        exercise_data = json_reader(fpath2 + '/exercise.json')
                        
                        # 1-4) heart_rate.json
                        try:
                            heart_rate_data = json_reader(fpath2 + '/resting_heart_rate.json')
                        except:
                            continue
                        
                        # 1-5) sleep_score.csv
                        # 1-6) sleep.json
                        sleep_data = json_reader(fpath2 + '/sleep.json')

                        # 1-7) steps.json
                        # 1-8) time_in_heart_rate_zones.json
                    
                    elif dir2 == 'pmsys':

                        wellness_data = csv_reader(fpath2 + "/wellness.csv")
                        wellness_dict = {k:[] for k in ['effective_time_frame', 'fatigue', 'mood', 'readiness', 'sleep_duration_h', 'sleep_quality', 'soreness', 'soreness_area', 'stress']}
                        for i,data in enumerate(wellness_data):
                            # ['effective_time_frame', 'fatigue', 'mood', 'readiness', 'sleep_duration_h', 'sleep_quality', 'soreness', 'soreness_area', 'stress']
                            if i == 0:
                                continue

                            date = data[0][:10] + "_" + data[0][11:][:-1].split(".")[0]
                            fatigue = data[1]
                            mood = data[2]
                            readiness = data[3]
                            sleep_dur = data[4]
                            sleep_qual = data[5]
                            # soreness = data[6]
                            # soreness_area = data[7]
                            stress = data[-1]

                            wellness_dict['effective_time_frame'].append(date)
                            wellness_dict['fatigue'].append(fatigue)
                            wellness_dict['mood'].append(mood)
                            wellness_dict['readiness'].append(readiness)
                            wellness_dict['sleep_duration_h'].append(sleep_dur)
                            wellness_dict['sleep_quality'].append(sleep_qual)
                            wellness_dict['stress'].append(stress)

                
                date = wellness_dict['effective_time_frame']
                fatigue = wellness_dict['fatigue']
                mood = wellness_dict['mood']
                readiness = wellness_dict['readiness']
                sleep_dur = wellness_dict['sleep_duration_h']
                sleep_qual = wellness_dict['sleep_quality']
                stress = wellness_dict['stress']

                for d,f,m,r,sd,sq,s in zip(date, fatigue, mood, readiness, sleep_dur, sleep_qual, stress):
                    # print("[INFO] Readiness:", r)
                    new_d = datetime.strptime(d, '%Y-%m-%d_%H:%M:%S')
                    # print("[INFO] target date:", new_d)

                    # [1] mood 
                    mood = m
                    stress = s
                    sleep_quality = sq
                    
                    # [2] exercise
                    exercise_hist = []
                    while True:
                        # TODO: datetime compare / 7-days or 14-days? -> refer to fitbit premium / heart rate
                        for e_data in exercise_data:
                            # print(e_data)
                            # print()
                            e_date = e_data['startTime'][:10] + "_" + e_data['startTime'][11:]
                            new_ed = datetime.strptime(e_date, '%Y-%m-%d_%H:%M:%S')
                            # print("[INFO] e_data's date:", new_ed)

                            # compare the date and make 2-weeks window
                            if (new_d > new_ed) and (new_d - new_ed) < timedelta(days=14):

                                try:
                                    activity = e_data['activityName']
                                    burn_calories = float(e_data['calories'])
                                    steps = float(e_data['steps'])
                                    duration = float(e_data['duration']) / 1000 / 60 # in minutes
                                    exercise_hist.append([new_ed, activity, duration, burn_calories, steps])
                                except:
                                    continue
                                
                        break                 
                    
                    # [3] sleep
                    sleep_hist = []
                    while True:
                        for s_data in sleep_data:
                            # print(s_data)
                            # print()
                            s_date = s_data['startTime'][:10] + "_" + s_data['startTime'][11:]
                            new_sd = datetime.strptime(s_date, '%Y-%m-%d_%H:%M:%S')

                            # compare the date and make 2-weeks window
                            if (new_d > new_sd) and (new_d - new_sd) < timedelta(days=14):
                                # print("[INFO2]", new_sd, "belongs")

                                sleep_duration = float(s_data['duration']) / 1000 / 60 # in minutes
                                sleep_hist.append([new_sd, sleep_duration])

                        break

                    # [4] heart rate
                    hr_hist = []
                    while True:
                        for hr_data in heart_rate_data:
                            # print(hr_data)
                            hr_date = hr_data['dateTime'][:10] + "_" + hr_data['dateTime'][11:]
                            new_hrd = datetime.strptime(hr_date, '%Y-%m-%d_%H:%M:%S')
                            # print("[INFO] e_data's date:", new_ed)

                            # compare the date and make 2-weeks window
                            if (new_d > new_hrd) and (new_d - new_hrd) < timedelta(days=14):
                                # print("[INFO3]", new_hrd, "belongs")
                                rhr = float(hr_data['value']['value'])
                                hr_hist.append([new_hrd, rhr])
                        break

                    # encode historical data
                    try:
                        steps_14d = sum([x[-1] for x in exercise_hist]) / len(exercise_hist)
                    except:
                        steps_14d = "N/A"
                        continue
                    try:
                        calories_14d = sum([x[-2] for x in exercise_hist]) / len(exercise_hist)
                    except:
                        calories_14d = "N/A"
                        continue
                    try:
                        rhr_14d = sum([x[-1] for x in hr_hist]) / len(hr_hist)
                    except:
                        rhr_14d = "N/A"
                        continue
                    try:
                        sleep_dur_14d = sum([x[-1] for x in sleep_hist]) / len(sleep_hist)
                    except:
                        sleep_dur_14d = "N/A"
                        continue

                    if SUBTASK == "readiness":
                        range1 = 0
                        range2 = 10
                    elif SUBTASK == "stress":
                        range1 = 1
                        range2 = 5
                    elif SUBTASK == "sleep_quality":
                        range1 = 1
                        range2 = 5 
                    elif SUBTASK == "fatigue":
                        range1 = 1
                        range2 = 5

                    I = "You are a personalized healthcare agent trained to predict {} which ranges from {} to {} based on physiological data and user information.".format(SUBTASK, range1, range2)        
                    Q = "The recent 14-days sensor readings show: [Steps]: {} steps, [Burned Calorories]: {} calories, [Resting Heart Rate]: {} beats/min, [SleepMinutes]: {} minutes, [Mood]: {} out of 5; What would be the predicted {}?".format([x[-1] for x in exercise_hist], [x[-2] for x in exercise_hist], [x[-1] for x in hr_hist], [x[-1] for x in sleep_hist], m, SUBTASK)

            
                    # NEW
                    if SUBTASK == "readiness":
                        # I = "Given the user's context and sensor data, predict the readiness score (ranges from 0 to 10)"
                        A = "The predicted {} level is {}.".format(SUBTASK, r)

                    elif SUBTASK == "stress":
                        # I = "Given the user's context and sensor data, predict the stress level (ranges from 1 to 5)"
                        A = "The predicted {} level is {}.".format(SUBTASK, s)

                    elif SUBTASK == "sleep_quality":
                        # I = "Given the user's context and sensor data, predict the sleep quality (ranges from 1 to 5)"
                        A = "The predicted {} level is {}.".format(SUBTASK, sq)

                    elif SUBTASK == "fatigue":
                        # I = "Given the user's context and sensor data, predict the fatigue score (ranges from 1 to 5)"
                        A = "The predicted {} level is {}.".format(SUBTASK, f)

                    final_data.append({'instruction':I, 'input':Q, 'output':A})
        
    elif DATA == "GLOBEM":
        SUBTASK = "anxiety" #"depression", "anxiety"
        print("[INFO] Dataset:", DATA)
        print("[INFO] Subtask:", SUBTASK)
        # TODO: add INS-W_2,3,4
        final_data = [] # json

        for insw_idx in tqdm([1,2,3,4]):
            data1 = csv_reader('medAlpaca/data/globem/INS-W_{}/SurveyData/dep_weekly.csv'.format(insw_idx))
            dict1 = {}
            keys1 = []

            for idx,_data in enumerate(data1):
                # ['', 'pid', 'date', 'feel_anxious', 'feel_depressed', 'BDI2', 'dep', 'dep_weekly_subscale', 'anx_weekly_subscale', 'dep_weeklysubscale_endterm_merged']
                if idx == 0:
                    for _idx,_d in enumerate(_data):
                        if _idx == 0:
                            continue
                        keys1.append(_d)
                        dict1[_d] = []
                else:
                    for _idx, _d in enumerate(_data):
                        dict1[keys1[_idx-1]].append(_d)
            
            data2 = csv_reader('medAlpaca/data/globem/INS-W_{}/FeatureData/steps.csv'.format(insw_idx))
            dict2 = {}
            keys2 = []

            for idx,_data in enumerate(data2):
                if idx == 0:
                    for _idx,_d in enumerate(_data):
                        if _idx == 0:
                            continue
                        keys2.append(_d)
                        dict2[_d] = []
                else:
                    for _idx, _d in enumerate(_data):
                        dict2[keys2[_idx-1]].append(_d)
            
            data3 = csv_reader('medAlpaca/data/globem/INS-W_{}/FeatureData/sleep.csv'.format(insw_idx))
            dict3 = {}
            keys3 = []

            for idx,_data in enumerate(data3):
                if idx == 0:
                    for _idx,_d in enumerate(_data):
                        if _idx == 0:
                            continue
                        keys3.append(_d)
                        dict3[_d] = []
                else:
                    for _idx, _d in enumerate(_data):
                        dict3[keys3[_idx-1]].append(_d)
            
            for idx,pid in enumerate(dict1['pid']):
                skip = False
                date = dict1['date'][idx]
                try:
                    dep = float(dict1['feel_depressed'][idx])
                    anx = float(dict1['feel_anxious'][idx])
                    # print("[{}]".format(idx))
                    # print("[dict1] pid: {}, date: {}, dep: {:.2f}".format(pid, date, dep))
                except:
                    continue
                
                i = 0
                while True:
                    if dict2['pid'][i] == pid and dict2['date'][i] == date:
                        interests = [
                            "f_steps:fitbit_steps_summary_rapids_maxsumsteps:14dhist", # The maximum daily step count during a time segment.
                            "f_steps:fitbit_steps_summary_rapids_minsumsteps:14dhist", 
                            "f_steps:fitbit_steps_summary_rapids_avgsumsteps:14dhist", 
                            "f_steps:fitbit_steps_summary_rapids_mediansumsteps:14dhist", 
                            "f_steps:fitbit_steps_summary_rapids_stdsumsteps:14dhist" 
                        ]
                        try:
                            maxsumsteps = float(dict2[interests[0]][i])
                            minsumsteps = float(dict2[interests[1]][i])
                            avgsumsteps = float(dict2[interests[2]][i])
                            mediansumsteps = float(dict2[interests[3]][i])
                            stdsumsteps = float(dict2[interests[4]][i])

                            # print("[dict2] pid: {}, date: {}, max: {:.2f}, min: {:.2f}, avg: {:.2f}, median: {:.2f}, std: {:.2f}".format(dict2['pid'][i], dict2['date'][i], maxsumsteps, minsumsteps, avgsumsteps, mediansumsteps, stdsumsteps))
                        except:
                            skip = True
                            break 
                        
                        break
                    else:
                        i += 1
                
                i = 0
                while True:
                    if dict3['pid'][i] == pid and dict3['date'][i] == date:
                        interests = [
                            "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:14dhist", # Average sleep efficiency for a certain sleep type during a time segment.
                            "f_slp:fitbit_sleep_summary_rapids_avgdurationafterwakeupmain:14dhist", # Average duration the user stayed in bed after waking up for a certain sleep type during a time segment.
                            "f_slp:fitbit_sleep_summary_rapids_avgdurationasleepmain:14dhist", # Average duration the user spent to fall asleep for a certain sleep type during a time segment.
                            "f_slp:fitbit_sleep_summary_rapids_avgdurationawakemain:14dhist", # Average duration the user stayed awake but still in bed for a certain sleep type during a time segment.
                            "f_slp:fitbit_sleep_summary_rapids_avgdurationtofallasleepmain:14dhist", # Average duration the user spent to fall asleep for a certain sleep type during a time segment.
                            "f_slp:fitbit_sleep_summary_rapids_avgdurationinbedmain:14dhist" 
                        ]
                        try:
                            durafwake = float(dict3[interests[1]][i])
                            dursleep = float(dict3[interests[2]][i])
                            durawake = float(dict3[interests[3]][i])
                            durfall = float(dict3[interests[4]][i])
                            durbed = float(dict3[interests[5]][i])
                            eff = float(dict3[interests[0]][i])

                            # print("[dict3] pid: {}, date: {}, eff: {:.2f}, durafwake: {:.2f}, dursleep: {:.2f}, durawake: {:.2f}, durfall: {:.2f}, durbed: {:.2f}".format(dict3['pid'][i], dict3['date'][i], eff, durafwake, dursleep, durawake, durfall, durbed))
                        except:
                            skip = True
                            break

                        break
                    else:
                        i += 1
                
                if skip == True:
                    continue

                # NEW
                I = "You are a personalized healthcare agent trained to predict PHQ-4 {} which ranges from 0 to 4 based on physiological data and user information.".format(SUBTASK)

                if SUBTASK == "depression":
                    # I = "Given the user's context and sensor data, predict the PHQ-4 depression level (from 0 to 4)"
                    A = "The predicted PHQ-4 {} score is {}.".format(SUBTASK, int(dep))
                elif SUBTASK == "anxiety":
                    # I = "Given the user's context and sensor data, predict the PHQ-4 anxiety level (from 0 to 4)"
                    A = "The predicted PHQ-4 {} score is {}.".format(SUBTASK, int(anx))
                
                Q = "The recent 14-days sensor readings show: [Steps] is {}. [Sleep] efficiency, duration the user stayed in bed after waking up, duration the user spent to fall asleep, duration the user stayed awake but still in bed, duration the user spent to fall asleep are {}, {}, {}, {}, {} mins in average; What would be the PHQ-4 {} score?".format(avgsumsteps, eff, durafwake, dursleep, durawake, durfall, durbed, SUBTASK)

                final_data.append({'instruction':I, 'input':Q, 'output':A})

    
    elif DATA == "MIT-BIH":
        SUBTASK = "ibis2a_fib"
        print("[INFO] Dataset:", DATA)
        print("[INFO] Subtask:", SUBTASK)
        final_data = []

        afib_data = json_reader('MIT-BIH/prompts.json')

        I = "You are a personalized healthcare agent trained to classify the given Interbeat Interval sequence in ms as either Atrial Fibrillation or Normal Sinus."
        
        for ad in afib_data:
            Q = ad['input']
            A = "The classified result is " + ad['output']
            tmp = {"insturction":I, 'input':Q, 'output':A}
            final_data.append(tmp)


    elif DATA == "MIMIC3":
        SUBTASK = "ibis2sinus_b" # "ibis2sinus_t"
        print("[INFO] Dataset:", DATA)
        print("[INFO] Subtask:", SUBTASK)
        final_data = []
        

        if SUBTASK == "ibis2sinus_b":
            sinus_data = json_reader('MIMIC3-WAVEFORM/sinus_bradycardia.json')
            I = "You are a personalized healthcare agent trained to classify the given Interbeat Interval sequence in ms as either Sinus Bradycardia or Normal Sinus."
        elif SUBTASK == "ibis2sinus_t":
            sinus_data = json_reader('MIMIC3-WAVEFORM/sinus_tachycardia.json')
            I = "You are a personalized healthcare agent trained to classify the given Interbeat Interval sequence in ms as either Sinus Tachycardia or Normal Sinus."

        for sd in sinus_data:
            Q = sd['input']
            A = "The classified result is " + sd['output']
            tmp = {"insturction":I, 'input':Q, 'output':A}
            final_data.append(tmp)



    # train/eval split
    N = len(final_data)
    final_train_data = []
    final_eval_data = []

    num_sd = {0:0, 1:0}
    num_sd_eval = {0:0, 1:0}


    import random
    random.seed(123)
    random.shuffle(final_data)
    eval_idx = 1
    for n,fd in enumerate(final_data):
        if n < int(N*0.5):
            if SUBTASK == "sleep_disorder":
                x = int(fd['output'].split()[-1].replace(".", ""))
                if x == 0 and num_sd[x] < 13:
                    final_train_data.append(fd)
                    num_sd[x] += 1
                elif x == 1 and num_sd[x] < 13:
                    final_train_data.append(fd)
                    num_sd[x] += 1
            
            elif "ibis" in SUBTASK:
                print(fd)
                x = fd['output']
                final_train_data.append(fd)


            else:
                final_train_data.append(fd)
        else:
            if SUBTASK == "sleep_disorder":
                
                if int(fd['output'].split()[-1].replace(".", "")) == 0 and num_sd_eval[int(fd['output'].split()[-1].replace(".", ""))] < 25:
                    x = int(fd['output'].split()[-1].replace(".", ""))
                    fd['no'] = eval_idx
                    fd['question'] = fd['input']
                    fd['answer'] = fd['output']
                    del fd['input']
                    del fd['output']

                    final_eval_data.append(fd)
                    eval_idx += 1
                    num_sd_eval[x] += 1
                
                elif int(fd['output'].split()[-1].replace(".", "")) == 1 and num_sd_eval[int(fd['output'].split()[-1].replace(".", ""))] < 25:
                    x = int(fd['output'].split()[-1].replace(".", ""))
                    fd['no'] = eval_idx
                    fd['question'] = fd['input']
                    fd['answer'] = fd['output']
                    del fd['input']
                    del fd['output']

                    final_eval_data.append(fd)
                    eval_idx += 1
                    num_sd_eval[x] += 1
                
            else:
                if eval_idx < 300:
                    fd['no'] = eval_idx
                    fd['question'] = fd['input']
                    fd['answer'] = fd['output']
                    del fd['input']
                    del fd['output']

                    final_eval_data.append(fd)
                    eval_idx += 1
        
    # 3-shot
    json_object_train_3 = json.dumps(final_train_data[:3], indent=4)
    # 10-shot
    json_object_train_10 = json.dumps(final_train_data[:10], indent=4)
    # 25-shot
    json_object_train_25 = json.dumps(final_train_data[:25], indent=4)
    json_object_eval = json.dumps(final_eval_data, indent=4)
    # fine-tune
    json_object_train_all = json.dumps(final_train_data, indent=4)

    # write
    # train
    with open("{}_{}_train_3.json".format(DATA, SUBTASK), "w") as outfile_train_3:
        outfile_train_3.write(json_object_train_3)
    with open("{}_{}_train_10.json".format(DATA, SUBTASK), "w") as outfile_train_10:
        outfile_train_10.write(json_object_train_10)
    with open("{}_{}_train_25.json".format(DATA, SUBTASK), "w") as outfile_train_25:
        outfile_train_25.write(json_object_train_25)
    with open("{}_{}_train_all.json".format(DATA, SUBTASK), "w") as outfile_train_all:
        outfile_train_all.write(json_object_train_all)
    # eval
    os.makedirs('../eval/data/{}_{}'.format(DATA.lower(), SUBTASK), exist_ok=True)
    with open("../eval/data/{}_{}/step1.json".format(DATA.lower(), SUBTASK), "w") as outfile_eval:
        outfile_eval.write(json_object_eval)


# n_samples = 100

# if MODE == "eval":
#     if DATA == "LifeSnaps":
#         life_snaps_data = json_reader("medAlpaca/data/LifeSnaps.json")
#         eval_data = []
#         no = 1
#         for data in life_snaps_data:
#             if random.random() < 0.06:
#                 tmp = {'no':no, 'question':data['input'], 'answer':data['output']}
#                 no += 1
#                 eval_data.append(tmp)
                
#         # serialize
#         json_object = json.dumps(eval_data, indent=4)
        
#         # write
#         with open("{}_eval.json".format(DATA), "w") as outfile:
#             outfile.write(json_object)
    
#     elif DATA == "":
#         pass

            
            


