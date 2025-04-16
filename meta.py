# from ray.rllib.policy import Policy

# policy_checkpoint_path = "E:\\agent\\policies"

# def load_policy(obs):
#     policy = Policy.from_checkpoint(policy_checkpoint_path)

#     action = policy.compute_single_action(obs)

#     return action


import os
import ray
import gym
import json
from ray.rllib.policy.policy import Policy

from arguments import get_args

from sklearn.metrics.pairwise import cosine_similarity
from embedding_encoder import get_embeddings_from_env

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


def load_model(obs, game_name, algo, is_need_compare=False):
    if not is_need_compare:
        plc = algo.get_policy()
        policy_ckpt = Policy.from_checkpoint(args.path + "/ckpts/" + game_name + "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())
        action = plc.compute_single_action(obs)
        return action[0]

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
            # print(pretty_print(result))

            if result["timesteps_total"] >= args.stop_timesteps:
                algo.save(checkpoint_dir=args.path + "/ckpts/" + curr_game_name + "/wholealgo")
                policy = algo.get_policy()
                policy.export_checkpoint(args.path + "/ckpts/" + curr_game_name + "/checkpoint")

                embedding_dict_str_keys = {str(k): v for k, v in embedding_dict.items()}
                write_buffer_data_json(args.path, embedding_dict_str_keys)

                action = policy.compute_single_action(obs)
                return action[0]

    