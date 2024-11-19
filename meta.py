import os
import ray
import gym
import csv
import json
import random
import numpy as np
from PIL import Image
from ray.rllib.policy.policy import Policy
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback

from arguments import get_args

from sklearn.metrics.pairwise import cosine_similarity
from embedding_encoder import get_embeddings_from_env
from task_mapper import train

args = get_args()

def read_buffer_data_json(directory):
    file_path = os.path.join(directory, "buffer_data.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data = {eval(k): v for k, v in data.items()}
    
    return data

def write_buffer_data_json(directory, data):
    file_path = os.path.join(directory, "buffer_data.json")
    
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Updated data written to {file_path}")

def find_similar_embedding(curr_image_embedding, curr_text_embedding, embedding_dict, threshold=0.9):
    best_weights_game = None
    highest_similarity = 0.0

    curr_image_embedding_flat = curr_image_embedding.flatten()
    curr_text_embedding_flat = curr_text_embedding.flatten()

    for key, game_name in embedding_dict.items():
        stored_image_embedding, stored_text_embedding = key
        
        image_similarity = cosine_similarity([curr_image_embedding_flat], [stored_image_embedding])[0][0]
        text_similarity = cosine_similarity([curr_text_embedding_flat], [stored_text_embedding])[0][0]

        combined_similarity = (image_similarity + text_similarity) / 2
        
        if combined_similarity >= threshold and combined_similarity > highest_similarity:
            highest_similarity = combined_similarity
            best_weights_game = game_name
    return best_weights_game

def cec_eval(obs, algo, env, game, threshold=0.8):
    trainer = train.FSCILTrainer(args)
    acc, class_idx, new_class, new_session = trainer.eval(obs)
    print('acc:', acc, 'class:', class_idx)

    if acc >= threshold:
        plc = algo.get_policy()
        policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + str(class_idx) + "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())
        action = plc.compute_single_action(obs)
        return action
    else:
        # start train specific policy
        policy = train_new_policy(algo, env, new_class, new_session, game, obs)

        args.pre_class = new_class
        # start incremental learning
        trainer.train(new_session)
        args.model_dir = f'/home/student/dell/checkpoint/atari/cec/session{new_session}_max_acc.pth'
        action = policy.compute_single_action(obs)
        return action

def save_observation_as_png(observation_list, save_dir, file_name_prefix):
    os.makedirs(save_dir, exist_ok=True)
    file_list = []

    for observation in observation_list:
        if len(observation.shape) == 3 and observation.shape[-1] == 4:
            for i in range(observation.shape[-1]):
                single_image = observation[:, :, i]
                
                image_data = (single_image * 255).astype(np.uint8)
                image = Image.fromarray(image_data)

                file_name = f"{file_name_prefix}_frame_{i}.png"
                image.save(os.path.join(save_dir, file_name))
                file_list.append(os.path.join(save_dir, file_name))
                print(f"Frame {i} saved as PNG: {os.path.join(save_dir, file_name)}")
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
    
    return file_list

def load_old_policy(algo, env, new_class, new_session, game, obs):
    plc = algo.get_policy()
    policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + str(new_class) + "/checkpoint")
    plc.set_weights(policy_ckpt.get_weights())
    policy = plc

    for _ in range(args.stop_timesteps):
        result = algo.train()
        # embed()
        print(pretty_print(result))

        if result["timesteps_total"] >= args.stop_timesteps:
            algo.save(checkpoint_dir=args.path + "/ckpts/" + str(new_class) + "/wholealgo")
            policy = algo.get_policy()
            policy.export_checkpoint(args.path + "/ckpts/" + str(new_class) + "/checkpoint")

            random_obs = env.get_random_obs()
            if obs is not None:
                image_name = f"obs_sample_{random.randint(1, 100)}"
                file_list = save_observation_as_png([obs], f'/home/student/dell/data/atari/{game}', image_name)
                create_new_session('/home/student/dell/data', new_class, new_session, file_list)

            return policy

def train_new_policy(algo, env, new_class, new_session, game, obs):
    has_Add_Session = False
    for _ in range(args.stop_timesteps):
        result = algo.train()
        # embed()
        print(pretty_print(result))

        if has_Add_Session == False:
            random_obs = env.get_random_obs()
            print('random_obs', len(random_obs))
            if obs is not None:
                image_name = f"obs_sample_{random.randint(1, 100)}"
                file_list = save_observation_as_png([obs], f'/home/student/dell/data/atari/{game}', image_name)
                create_new_session('/home/student/dell/data', new_class, new_session, file_list)
                has_Add_Session = True


        if result["timesteps_total"] >= args.stop_timesteps:
            algo.save(checkpoint_dir=args.path + "/ckpts/" + str(new_class) + "/wholealgo")
            policy = algo.get_policy()
            policy.export_checkpoint(args.path + "/ckpts/" + str(new_class) + "/checkpoint")

            # random_obs = env.get_random_obs()
            # print('random_obs', len(random_obs))
            # if len(random_obs) != 0:
            #     image_name = f"obs_sample_{random.randint(1, 100)}"
            #     file_list = save_observation_as_png(random_obs, f'/home/student/dell/data/atari/{game}', image_name)
            #     create_new_session('/home/student/dell/data', new_class, new_session, file_list)

            # embedding_dict_str_keys = {str(k): v for k, v in embedding_dict.items()}
            # write_buffer_data_json(args.path, embedding_dict_str_keys)

            return policy

def create_new_session(data_root, new_class, new_session, images_to_add):
    # max_number = self.get_cur_session_num(data_root)

    new_session_file = f'session_{new_session}.txt'
    new_session_path = os.path.join(data_root, new_session_file)

    mid_index = len(images_to_add) // 2
    with open(new_session_path, 'w') as f:
        # f.write(f'{images_to_add[0]}\n')
        for image in images_to_add[:mid_index]:
            f.write(f'{image}\n')
    print(f'create new session file: {new_session_file}')

    train_csv_path = os.path.join(data_root, 'train.csv')
    test_csv_path = os.path.join(data_root, 'test.csv')
    train_txt_path = os.path.join(data_root, 'train.txt')
    test_txt_path = os.path.join(data_root, 'test.txt')

    labels_to_add = [new_class] * len(images_to_add)
    train_images = images_to_add[:mid_index]
    test_images = images_to_add[mid_index:]
    train_labels = labels_to_add[:mid_index]
    test_labels = labels_to_add[mid_index:]

    # update train.csv
    with open(train_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for image, label in zip(train_images, train_labels):
            writer.writerow([image, label])
    print(f'update {train_csv_path}')

    # update test.csv
    with open(test_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for image, label in zip(test_images, test_labels):
            writer.writerow([image, label])
    print(f'update {test_csv_path}')

    # update train.txt
    with open(train_txt_path, 'a') as f:
        for image in train_images:
            f.write(f'{image}\n')
    print(f'update {train_txt_path}')

    # update test.txt
    with open(test_txt_path, 'a') as f:
        for image in test_images:
            f.write(f'{image}\n')
    print(f'update {test_txt_path}')

def load_model(obs, game_name, algo, env, is_need_compare=False):
    if not is_need_compare:
        plc = algo.get_policy()
        if args.pre_class is not None:
            policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + str(args.pre_class) + "/checkpoint")
        else:
            policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + game_name + "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())
        action = plc.compute_single_action(obs)
        return action[0]


    if args.mapper_mode == 'cec':
        # compare with cec task-mapper
        action = cec_eval(obs, algo, env, game_name)
        return action[0]
    else:
        # compare with cos similarity
        curr_image_embedding, curr_text_embedding = get_embeddings_from_env(obs, game_name, is_random=True)
        embedding_dict = read_buffer_data_json(args.path)

        similar_weights_game = find_similar_embedding(curr_image_embedding, curr_text_embedding, embedding_dict)

        plc = algo.get_policy()

        curr_game_name = game_name
        if similar_weights_game is not None:
            print("Found similar embeddings, using previous policy weights.")
            curr_game_name = similar_weights_game
            policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + curr_game_name + "/checkpoint")
            plc.set_weights(policy_ckpt.get_weights())
            action = plc.compute_single_action(obs)
            return action[0]
        else:
            print("No similar embeddings found, training new policy.")
            embedding_key = (tuple(curr_image_embedding.flatten()), tuple(curr_text_embedding.flatten()))
            embedding_dict[embedding_key] = game_name

            for _ in range(args.stop_timesteps):
                result = algo.train()
                # embed()
                print(pretty_print(result))

                if result["timesteps_total"] >= args.stop_timesteps:
                    algo.save(checkpoint_dir=args.path + "/ckpts/" + curr_game_name + "/wholealgo")
                    policy = algo.get_policy()
                    policy.export_checkpoint(args.path + "/ckpts/" + curr_game_name + "/checkpoint")

                    embedding_dict_str_keys = {str(k): v for k, v in embedding_dict.items()}
                    write_buffer_data_json(args.path, embedding_dict_str_keys)

                    action = policy.compute_single_action(obs)
                    return action[0]

    