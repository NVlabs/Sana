import numpy as np


def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape


def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """
    Store rows of multiple numpy arrays in a single LMDB.
    Each row is stored separately with a naming convention.
    """
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                # Convert row to bytes
                if isinstance(row, str):
                    row_bytes = row.encode()
                else:
                    row_bytes = row.tobytes()

                data_key = f"{array_name}_{start_index + i}_data".encode()

                txn.put(data_key, row_bytes)


def process_data_dict(data_dict, seen_prompts):
    output_dict = {}

    all_videos = []
    all_prompts = []
    for prompt, video in data_dict.items():
        if prompt in seen_prompts:
            continue
        else:
            seen_prompts.add(prompt)

        video = video.half().numpy()
        all_videos.append(video)
        all_prompts.append(prompt)

    if len(all_videos) == 0:
        return {"latents": np.array([]), "prompts": np.array([])}

    all_videos = np.concatenate(all_videos, axis=0)

    output_dict["latents"] = all_videos
    output_dict["prompts"] = np.array(all_prompts)

    return output_dict


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f"{array_name}_{row_index}_data".encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array
