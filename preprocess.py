from file_util import *
from midi_util import *
# from main import *

classical_raw_path = "./dataset/classical"
jazz_raw_path = "./dataset/jazz"
quant = 2

if not os.path.exists(classical_raw_path + '_valid'):
    validate_data(classical_raw_path, quant)
if not os.path.exists(jazz_raw_path + '_valid'):
    validate_data(jazz_raw_path, quant)
if not os.path.exists(classical_raw_path + '_valid_quantized'):
    quantize_data(classical_raw_path + '_valid', quant)
if not os.path.exists(jazz_raw_path + '_valid_quantized'):
    quantize_data(jazz_raw_path + '_valid', quant)

if not os.path.exists(classical_raw_path + '_valid_quantized_inputs'):
    save_data(classical_raw_path + '_valid_quantized', quant)
if not os.path.exists(jazz_raw_path + '_valid_quantized_inputs'):
    save_data(jazz_raw_path + '_valid_quantized', quant)

data_for_model = "./data_for_model"
if not os.path.exists(data_for_model):
    os.makedirs(data_for_model)

jazz_file_names = []
classical_file_names = []

for i, filename in enumerate(os.listdir(classical_raw_path + '_valid_quantized_inputs')):
    classical_file_names.append(filename)

for i, filename in enumerate(os.listdir(jazz_raw_path + '_valid_quantized_inputs')):
    jazz_file_names.append(filename)

import random

random.shuffle(classical_file_names)
random.shuffle(jazz_file_names)

train_count, valid_count = int(0.7 * len(classical_file_names)), int(0.15 * len(classical_file_names))

train_classical_filenames = classical_file_names[:train_count]
valid_classical_filenames = classical_file_names[train_count:train_count + valid_count]
test_classical_filenames = classical_file_names[train_count + valid_count:]

train_count, valid_count = int(0.7 * len(jazz_file_names)), int(0.15 * len(jazz_file_names))

train_jazz_filenames = jazz_file_names[:train_count]
valid_jazz_filenames = jazz_file_names[train_count:train_count + valid_count]
test_jazz_filenames = jazz_file_names[train_count + valid_count:]

from shutil import copyfile

if not os.path.exists(data_for_model + '/inputs'):
    os.makedirs(data_for_model + '/inputs')
    os.makedirs(data_for_model + '/inputs/classical')
    os.makedirs(data_for_model + '/inputs/jazz')

    for each_file in train_classical_filenames:
        copyfile(classical_raw_path + '_valid_quantized_inputs/' + each_file,
                 data_for_model + '/inputs/classical/' + each_file)
    for each_file in train_jazz_filenames:
        copyfile(jazz_raw_path + '_valid_quantized_inputs/' + each_file, data_for_model + '/inputs/jazz/' + each_file)

if not os.path.exists(data_for_model + '/velocities'):
    os.makedirs(data_for_model + '/velocities')
    os.makedirs(data_for_model + '/velocities/classical')
    os.makedirs(data_for_model + '/velocities/jazz')

    for each_file in train_classical_filenames:
        copyfile(classical_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/velocities/classical/' + each_file)
    for each_file in train_jazz_filenames:
        copyfile(jazz_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/velocities/jazz/' + each_file)

if not os.path.exists(data_for_model + '/test'):
    os.makedirs(data_for_model + '/test')
    os.makedirs(data_for_model + '/test/inputs/classical')
    os.makedirs(data_for_model + '/test/inputs/jazz')
    os.makedirs(data_for_model + '/test/velocities/classical')
    os.makedirs(data_for_model + '/test/velocities/jazz')

    for each_file in test_classical_filenames:
        copyfile(classical_raw_path + '_valid_quantized_inputs/' + each_file,
                 data_for_model + '/test/inputs/classical/' + each_file)
        copyfile(classical_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/test/velocities/classical/' + each_file)

    for each_file in test_jazz_filenames:
        copyfile(jazz_raw_path + '_valid_quantized_inputs/' + each_file,
                 data_for_model + '/test/inputs/jazz/' + each_file)
        copyfile(jazz_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/test/velocities/jazz/' + each_file)

if not os.path.exists(data_for_model + '/eval'):
    os.makedirs(data_for_model + '/eval')
    os.makedirs(data_for_model + '/eval/inputs/classical')
    os.makedirs(data_for_model + '/eval/inputs/jazz')
    os.makedirs(data_for_model + '/eval/velocities/classical')
    os.makedirs(data_for_model + '/eval/velocities/jazz')

    for each_file in valid_classical_filenames:
        copyfile(classical_raw_path + '_valid_quantized_inputs/' + each_file,
                 data_for_model + '/eval/inputs/classical/' + each_file)
        copyfile(classical_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/eval/velocities/classical/' + each_file)

    for each_file in valid_jazz_filenames:
        copyfile(jazz_raw_path + '_valid_quantized_inputs/' + each_file,
                 data_for_model + '/eval/inputs/jazz/' + each_file)
        copyfile(jazz_raw_path + '_valid_quantized_velocities/' + each_file,
                 data_for_model + '/eval/velocities/jazz/' + each_file)