import numpy as np
import actor
import learner
import os

_NUM_ACTORS = 10
_BATCH_SIZE = 500
_SAVE_GAP = 50

_FILE_PATH = os.getcwd() + '/data/'

def generate_actor_config(batch_size, num_actors):
    actor_config = actor.RNaDConfig(
        game_name='junqi1',
        trajectory_max=40,
        state_representation=actor.StateRepresentation.OBSERVATION,
        policy_network_layers=(16, 16, 16, 16),
        batch_size=batch_size // num_actors,
        learning_rate=0.00005,
        adam=actor.AdamConfig(),
        clip_gradient=10_000,
        target_network_avg=0.001,
        entropy_schedule_repeats=(1,),
        entropy_schedule_size=(20_000,),
        eta_reward_transform=0.2,
        nerd=actor.NerdConfig(),
        c_vtrace=1.0,
        finetune=actor.FineTuning(),
        seed=np.random.randint(1, 101)
    )
    
    return actor_config

def generate_learner_config(batch_size):
    learner_config = actor.RNaDConfig(
        game_name='junqi1',
        trajectory_max=40,
        state_representation=learner.StateRepresentation.OBSERVATION,
        policy_network_layers=(16, 16, 16, 16),
        batch_size=batch_size,
        learning_rate=0.00005,
        adam=learner.AdamConfig(),
        clip_gradient=10_000,
        target_network_avg=0.001,
        entropy_schedule_repeats=(1,),
        entropy_schedule_size=(20_000,),
        eta_reward_transform=0.2,
        nerd=learner.NerdConfig(),
        c_vtrace=1.0,
        finetune=learner.FineTuning(),
        seed=42,
    )
    
    return learner_config