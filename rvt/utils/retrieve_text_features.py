import tqdm
from rvt.utils.get_dataset import get_dataset
from rvt.utils.rvt_utils import RLBENCH_TASKS
import pdb
import json
import torch
import clip
import os
import pickle

# get_dataset_func = lambda: get_dataset(
#         RLBENCH_TASKS,
#         1,
#         None,
#         "/mnt/ssd1/telima/datasets/rvt_replay",
#         None,
#         './',
#         100,
#         None,
#         False,
#         'cuda:1',
#         num_workers=1,
#         only_train=True,
#         sample_distribution_mode='transition_uniform',
#     )
    
# train_dataset, _ = get_dataset_func()
# data_iter = iter(train_dataset)

# iter_command = range(240000)
# rank = 0
# out = []
# for iteration in tqdm.tqdm(
#     iter_command, disable=(rank != 0), position=0, leave=True
# ):
#     raw_batch = next(data_iter)
#     lang_goal = raw_batch['lang_goal'][0][0]

#     if lang_goal not in out:
#         out.append(lang_goal)
#         print(lang_goal)

# with open('./text1.json', 'w') as f:
#     json.dump(out, f)

# f.close()

##############################################################
# EPISODE_FOLDER = "episode%d"
# VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"
# from rvt.utils.rvt_utils import RLBENCH_TASKS

# data_path = '/mnt/ssd1/telima/datasets/RLBench/'
# out = []
# for task in RLBENCH_TASKS:
#     EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"

#     for d_idx in range(100):
#         varation_descs_pkl_file = os.path.join(
#             data_path, EPISODES_FOLDER_TRAIN, EPISODE_FOLDER%d_idx, VARIATION_DESCRIPTIONS_PKL
#         )
#         with open(varation_descs_pkl_file, "rb") as f:
#             descs = pickle.load(f)
        
#         desc = descs[0]
#         if desc not in out:
#             out.append(desc)
# print(len(out))
# with open('./text1.json', 'w') as f:
#     json.dump(out, f)

##############################################################
device = 'cuda:1'
clip_model, clip_preprocess = clip.load("RN50", device=device)
clip_model.eval()

def clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
    # x = x @ clip_model.text_projection

    return x, emb

res = json.load(open('./text.json'))

tokens = clip.tokenize(res).numpy()
token_tensor = torch.from_numpy(tokens).to(device)
with torch.no_grad():
    lang_feats, lang_embs = clip_encode_text(clip_model, token_tensor)
pdb.set_trace()
lang_dict = {}
lang_dict['lang_token'] = lang_feats.float()
lang_dict['lang_embs'] = lang_embs.float()

# torch.save(lang_dict, 'lang2.pth')


