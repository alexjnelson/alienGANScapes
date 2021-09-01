# this script will run training on each model in ./tests/models. the checkpoints and sample outputs of each model will be saved in
# tests/outputs in a subfolder with the same name as the respective model script

import os


PATH_TO_TESTS = './tests'
datapath = './images'


EPOCHS = 50
BATCH_SIZE = 128

path_to_models = os.path.join(PATH_TO_TESTS, 'models')

# for model in os.listdir(path_to_models):
#     if model == 'dataset.py':
#         continue
#     modelpath = os.path.join(path_to_models, model)
#     # before training the models, make sure each model's tensor flows are working
#     os.system(f'python "{modelpath}" -m 0')

for model in os.listdir(path_to_models):
    # don't execute dataset.py
    if model == 'dataset.py':
        continue
    modelpath = os.path.join(path_to_models, model)
    # determine paths for model's output
    modelout = os.path.join(PATH_TO_TESTS, 'outputs', model.replace('.py', ''))
    samplepath = os.path.join(modelout, 'samples')
    checkpointpath = os.path.join(modelout, 'checkpoints')
    # create output directories
    try:
        os.mkdir(modelout)
    except FileExistsError:
        pass
    try:
        os.mkdir(samplepath)
    except FileExistsError:
        pass
    try:
        os.mkdir(checkpointpath)
    except FileExistsError:
        pass
    # run model so it outputs in the right directories
    # os.system(f'python "{modelpath}" -m 1 -c "{checkpointpath}" -s "{samplepath}" -d "{datapath}"')
    os.system(f'python "{modelpath}" -m 2 -c "{checkpointpath}" -s "{samplepath}" -d "{datapath}"')
