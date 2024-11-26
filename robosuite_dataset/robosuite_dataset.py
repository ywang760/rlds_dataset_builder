from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py

class RobosuiteDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Full train and valid splits',
      '1.0.1': '20_percent_train and 20_percent_valid splits',
      '1.0.2': '50_percent_train and 50_percent_valid splits'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        dataset_dir = "/users/ywang760/scratch/robomimic/datasets/lift/ph"
        image_dataset_path = dataset_dir + "/image_v141.hdf5"
        low_dim_dataset_path = dataset_dir + "/low_dim_v141.hdf5"
        self.dataset = h5py.File(image_dataset_path, 'r')

        # TODO: modify this based on dataset
        self.language_instruction = "Lift the red cube"
        self.language_embedding = self._embed([self.language_instruction])[0].numpy()
        
        # TODO: modify the mask
        self.prefix = "" # or "50_percent_" or "20_percent_"
        

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(84, 84, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'object': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float32,
                            doc='Object state.',
                        ),
                        'eef_pos': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='End effector position.',
                        ),
                        'eef_quat': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='End effector quaternion.',
                        ),
                        'eye_in_hand_image': tfds.features.Image(
                            shape=(84, 84, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Eye in hand camera RGB observation.',
                        ),
                        'gripper_qpos': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='Gripper position.',
                        ),
                        'gripper_qvel': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='Gripper velocity.',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint positions.',
                        ),
                        'joint_pos_cos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint positions cosine.',
                        ),
                        'joint_pos_sin': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint positions sine.',
                        ),
                        'joint_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint velocities.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'state': tfds.features.Tensor(
                        shape=(32,),
                        dtype=np.float32,
                        doc='State representation.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step where done is marked as True.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'original_name': tfds.features.Text(
                        doc='Key name in the original data file, e.g. demo_20.'
                    ),
                    'length': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Number of steps in the episode.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        train_mask_name = f"{self.prefix}train"
        val_mask_name = f"{self.prefix}valid"
        train_mask = self.dataset["mask"][train_mask_name][:]
        val_mask = self.dataset["mask"][val_mask_name][:]
        train_keys = set([name.decode('utf-8') for name in train_mask])
        val_keys = set([name.decode('utf-8') for name in val_mask])

        return {
            'train': self._generate_examples(keys=train_keys),
            'val': self._generate_examples(keys=val_keys),
        }

    def _generate_examples(self, keys) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(sample):
            # load raw data --> this should change for your dataset
            data = self.dataset["data"][sample]
            
            episode = []
            length = data["actions"].shape[0] + 1

            # figure out the terminal index
            terminal_index = data["dones"][...].tolist().index(1)

            for index in range(length):
                obs = data['obs'] if index < length - 1 else data['next_obs']
                i = index if index < length - 1 else -1
                step = {
                    'observation': {
                        'image': obs['agentview_image'][i], # shape (84, 84, 3)
                        'object': obs['object'][i].astype(np.float32), # shape (10, )
                        'eef_pos': obs['robot0_eef_pos'][i].astype(np.float32), # shape (3, )
                        'eef_quat': obs['robot0_eef_quat'][i].astype(np.float32), # shape (4, )
                        'eye_in_hand_image': obs['robot0_eye_in_hand_image'][i], # shape (84, 84, 3)
                        'gripper_qpos': obs['robot0_gripper_qpos'][i].astype(np.float32), # shape (2, )
                        'gripper_qvel': obs['robot0_gripper_qvel'][i].astype(np.float32), # shape (2, )
                        'joint_pos': obs['robot0_joint_pos'][i].astype(np.float32), # shape (7, )
                        'joint_pos_cos': obs['robot0_joint_pos_cos'][i].astype(np.float32), # shape (7, )
                        'joint_pos_sin': obs['robot0_joint_pos_sin'][i].astype(np.float32), # shape (7, )
                        'joint_vel': obs['robot0_joint_vel'][i].astype(np.float32), # shape (7, )
                    },
                    'action': data['actions'][i].astype(np.float32), # shape (7, )
                    'discount': 1.0,
                    'reward': data['rewards'][i].astype(np.float32),
                    'state': data['states'][i].astype(np.float32), # shape (32, )
                    'is_first': index == 0,
                    'is_last': index == length - 1,
                    'is_terminal': index == terminal_index,
                    'language_instruction': self.language_instruction,
                    'language_embedding': self.language_embedding,
                }
                episode.append(step)
            

            assert len(episode) == length, f"Length mismatch: {len(episode)} != {length}"
            # create output data sample
            output = {
                'steps': episode,
                'episode_metadata': {
                    'original_name': sample,
                    'length': length,
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return sample, output

        # for smallish datasets, use single-thread parsing
        for i, sample in enumerate(keys):
            print(f"Processing sample {i + 1}/{len(keys)}")
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

