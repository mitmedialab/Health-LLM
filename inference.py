import os
import random
import numpy as np
import json
from tqdm import tqdm

import openai
import google.generativeai as genai

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
from transformers import AutoModelForQuestionAnswering


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5')
# parser.add_argument('--dataset', type=str, default='pmdata')
args = parser.parse_args()

model = args.model
# dataset = args.dataset

openai.api_key = os.environ["openai_key"]
genai.configure(api_key=os.environ["genai_key"])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def get_response(model, question, seed):
    if model == 'gemini-pro':
        config = {
            "max_output_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1
        }

        chat = model.start_chat()
        return chat.send_message(question, generation_config=config).text
    
    elif model == "gpt-4":
        messages = [{"role": "user", "content": question}]

        functions = [
            {
                "name": "predict health measure",
                "description": "Predict the health measure with given physiological data and demographic context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "physiological": {
                            "type": "string",
                            "description": "The physiological data, e.g. number of steps, heart rate, calorie burn, mood",
                        },
                        "unit": {"type": "string", "enum": ["score", "level"]},
                    },
                    "required": ["physiological"],
                },
            }
        ]

        url = 'https://api.openai.com/v1/chat/completions'
        headers = {"Authorization": "Bearer {}".format("")}
        data = {'model': 'gpt-4', 'messages': messages}
        response = requests.post(url, headers=headers, json=data).json()

        return response
    
    elif model == 'gpt-3.5':
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=question,
            max_tokens=120
        )

        return response.choices[0].text.strip()

    elif "medAlpaca" in model:
        try:
            ans = medalpaca_pl(question)['generated_text']
        
        except Exception as e:
            ans = "N/A"
        
        return ans


for mode in ['zero-shot', 'few-shot', 'few-shot_cot', 'few-shot_cot-sc']:
    print("Mode:", mode)

    data_task_dict = {'pmdata':['fatigue', 'stress', 'readiness', 'sleep_quality'], 'lifesnaps':['stress_resilience', 'sleep_disorder'], 'globem': ['anxiety', 'depression'], 'awfb':['calories', 'activity'], 'mimic3':['ibis2sinus_b', 'ibis2sinus_t'], 'mit-bih':['ibis2a_fib']} 
    for dataset,tasks in data_task_dict.items():
        print("|__Dataset:", dataset)
        for task in tasks:
            print("|___Subtask:", task)
        
            for seed in [0, 1, 2]:
                print("|____Seed:", seed)
                
                if os.path.exists("output/gemini-pro/{}/{}_sd{}.json".format(mode, task, seed)):
                    print("skipping ...")
                    continue

                data = json_reader('zero-shot/data/{}/{}.json'.format(dataset, task))
           
                res = []
                num_samples = 0
                for _data in tqdm(data):
                    question = _data['input']
                    answer = _data['output'] 
 
                    if task == 'stress':
                        rand_num = random.choice([0,1,2,3,4,5])
                    elif task == 'stress_resilience':
                        rand_num = float("{:.2f}".format(random.uniform(0,5)))
                    elif task == 'depression':
                        rand_num = random.choice([0,1,2,3,4])
                    elif task == 'anxiety':
                        rand_num = random.choice([0,1,2,3,4])
                    elif task == 'readiness':
                        rand_num = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    elif task == 'fatigue':
                        rand_num = random.choice([0,1,2,3,4,5])
                    elif task == 'activity':
                        rand_num = random.choice(['Running 7 METs', 'Self Pace walk', 'Sitting', 'Lying', 'Running 3 METs', 'Running 5 METs'])
                    elif task == 'calories':
                        rand_num = float("{:.2f}".format(random.uniform(0, 100)))
                    elif task == 'sleep_quality':
                        rand_num = random.choice([1,2,3,4,5])
                    elif task == 'sleep_disorder':
                        rand_num = random.choice([0, 1])
                    elif task == 'ibis2a_fib':
                        rand_num = random.choice(['Normal Sinus', 'Atrial Fibrillation'])
                    elif task == 'ibis2sinus_b':
                        rand_num = random.choice(['Sinus Bradycardia', 'Normal Sinus'])
                    elif task == 'ibis2sinus_t':
                        rand_num = random.choice(['Normal Sinus', 'Sinus Tachycardia'])

                    format_prompt = "For example, the answer should be in the following format:\nAnswer: {}".format(rand_num)

                    if mode == 'zero-shot':
                        instruction = "You are a health assistant. Your mission is to read the following input health query and return your prediction.\n"
                        question = instruction + question + format_prompt
                    
                    elif mode == 'few-shot':
                        instruction = "You are a health assistant. Your mission is to read the following examplers and return your prediction based on the health query\n"
                        examplers = ""
                        N = 3

                        for ei in range(N):
                            idx = random.choice([i for i in range(len(data))][50:])
                            example_question = data[idx]['input']
                            example_answer = data[idx]['output'] 

                            examplers += "\n[Example {}]\n{}\nAnswer: {}\n".format(ei+1, example_question, example_answer)

                        question = instruction + examplers + "\nFinally, please answer to the below question:\n" + question + format_prompt

                    elif mode == 'few-shot_cot':
                        instruction = "You are a health assistant. Your mission is to read the following examplers and return your prediction based on the health query\n"
                        examplers = ""
                        N = 3

                        for ei in range(N):
                            idx = random.choice([i for i in range(len(data))][50:])

                            example_question = data[idx]['input']
                            example_answer = data[idx]['output'] 

                            cot_prompt = "You are a health assistant. Your mission is to read the given health query with the label and then return an answer with short explanation that's supporting the label.\nQuestion: {}\nLabel: {}.".format(example_question, example_answer)

                            while True:
                                try:
                                    model = genai.GenerativeModel('gemini-pro')
                                    explanation = get_response(model, cot_prompt, seed)
                                    break
                                except Exception as e:
                                    print(e)
                                    count +=1
                                    continue

                            examplers += "\n[Example {}]\n{}\nExplanation: {}\nAnswer: {}\n".format(ei+1, example_question, explanation, example_answer)

                        question = instruction + examplers + "\nFinally, please answer to the below question:\n" + question + format_prompt

                    elif mode == 'few-shot_cot-sc':
                        instruction = "You are a health assistant. Your mission is to read the following examplers and return your prediction based on the health query\n"
                        examplers = ""
                        N = 3
                        
                        for ei in range(N):
                            idx = random.choice([i for i in range(len(data))][50:])
                        
                            example_question = data[idx]['input']
                            example_answer = data[idx]['output'] 

                            cot_prompt = "You are a health assistant. Your mission is to read the given health query with the label and then return an answer with short explanation that's supporting the label.\nQuestion: {}\nLabel: {}.".format(example_question, example_answer)

                            while True:
                                try:
                                    model = genai.GenerativeModel('gemini-pro')
                                    explanation = get_response(model, cot_prompt, seed)
                                    break
                                except Exception as e:
                                    print(e)
                                    continue
                            
                            examplers += "\n[Example {}]\n{}\nExplanation: {}\nAnswer: {}\n".format(ei+1, example_question, explanation, example_answer)

                        question = instruction + examplers + "\nFinally, please answer to the below question:\n" + question + format_prompt

                    if mode != 'few-shot_cot-sc':
                        while True:
                            try:
                                model = genai.GenerativeModel('gemini-pro')
                                response = get_response(model, question, seed)
                                break
                            except Exception as e:
                                print(e)
                                continue
                        
                        print("question:", question)
                        print("response:", response)
                        print("label:", answer)
                        print()
                        res.append({"no": num_samples + 1, "question": question, "answer": response, 'label': answer})
                    
                    else:
                        responses = []
                        for _ in range(5):
                            while True:
                                try:
                                    model = genai.GenerativeModel('gemini-pro')
                                    response = get_response(model, question, seed)
                                    break
                                except Exception as e:
                                    print(e)
                                    continue
                            
                            responses.append(response)
                        
                        print("question:", question)
                        
                        for _response in responses:
                            print("response:", _response)
                            print()
                        
                        print("label:", answer)
                        print()
                        res.append({"no": num_samples + 1, "question": question, "answer": responses, 'label': answer})

                    num_samples += 1
                
                json_object = json.dumps(res, indent=4)
                with open("output/gemini-pro/{}/{}_sd{}.json".format(mode, task, seed), "w") as outfile:
                    outfile.write(json_object)
