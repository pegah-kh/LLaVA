import json
import os
import pickle

with open(os.path.join('/home/khayatan/llava/prepare_data', 'blip_laion_cc_sbu_558k.json'), 'r') as f:


    question_conversations = json.load(f)


set_random_image_questions = set()
for item in question_conversations:
    current_random_question = item['conversations'][0]['value']
    set_random_image_questions.add(current_random_question)



with open(os.path.join('/home/khayatan/llava/prepare_data', 'llava_random_image_questions.pkl'), 'wb') as f:
    list_random_image_questions = list(set_random_image_questions)
    dict_llava_image_questions = {'image_questions': list_random_image_questions}


    pickle.dump(dict_llava_image_questions, f)