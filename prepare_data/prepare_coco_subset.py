import pickle
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader


import os, sys
sys.path.append(os.path.abspath('/home/khayatan/llava/'))
from dataset.caption import get_loader

data_dir="/data/mshukor/data/vl_adapter/vlt5_dataset"



train_loader, train_dataset  = get_loader(
            split='train', mode='train', batch_size=2,
            distributed=False,
            workers=4,
            topk=-1,
            data_dir=data_dir,
            local_rank=None, world_size=None, verbose=True,
            image_size=224, use_data_augmentation=False, 
        )




import pickle

with open('/home/khayatan/finetune_DePalm/ep-alm/coco_subsets/sample_noun_idx_coco_actions_extended.pkl', 'rb') as handle:
    select_id_action_2 = pickle.load(handle)

num_samp = {key: len(select_id_action_2[key]) for key in select_id_action_2.keys()}
sum_ = sum([len(select_id_action_2[key]) for key in select_id_action_2.keys()])
print(sum_)



with open(os.path.join('/home/khayatan/llava/prepare_data/', 'llava_random_image_questions.pkl'), 'rb') as f:

    llava_image_questions_dict = pickle.load(f)


llava_image_questions = llava_image_questions_dict['image_questions']
samples_for_finetuning = []



data_dir='/data/mshukor/data'
dataset_dir = Path(data_dir)
coco_dir = dataset_dir.joinpath('COCO')
coco_img_dir = coco_dir.joinpath('images/')

source_to_h5 = {}

source_to_h5.update({
    'train2014': coco_img_dir.joinpath(f'train2014'),
    'val2014': coco_img_dir.joinpath(f'val2014'),
})



for key, indices in select_id_action_2.items():
    # accumulate the samples like:
    """
    [
    {
        "id": "997bb945-628d-4724-b370-b84de974a19f",
        "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
        },
        {
            "from": "gpt",
            "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
        },
        ]
    },
    ...
    ]
    """
    # randomly choosing image questions from the llava used questions
    # llava_question_indices = random.sample(range(len(llava_image_questions)), len(indices))
    llava_question_indices = random.choices(range(len(llava_image_questions)), k=len(indices))

    for i, index in enumerate(indices):
        sample = train_dataset[index]
        id_ = sample['img_id']


        if 'train' in id_:
            source = 'train2014'
        elif 'val' in id_:
            source = 'val2014'
        
        path = str(source_to_h5[source].joinpath(f"{id_}.jpg"))



        ############################################
        image_ = path

        ############################################
        

        conversations_ = [
        {
            "from": "human",
            "value": llava_image_questions[llava_question_indices[i]]
        },
        {
            "from": "gpt",
            "value": sample['sent']
        },
        ]

        llava_format_item = {
            "id": id_,
            "image": image_,
            "conversations": conversations_
        }


        # print(llava_format_item)



        samples_for_finetuning.append(llava_format_item)




with open(os.path.join('/home/khayatan/llava/prepare_data', 'llava_coco_actions.json'), 'w') as f:

    json.dump(samples_for_finetuning, f, indent=4)






