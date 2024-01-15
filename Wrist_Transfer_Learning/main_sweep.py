# Import the W&B Python Library and log into W&B
import wandb
import os
from model_pipeline import model_pipeline, accuracy_data

wandb.login()

# os.chdir('/home/sharonlabgpu/noa_and_hadar/just_preprocess')


def main():
    wandb.init()
    test_acc = model_pipeline(wandb.config)
    wandb.log({"Test accuracy": test_acc})


# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'Test accuracy'},
    'parameters':
        {'epochs': {'values': [5, 10, 15]},
         'num_of_layers_unfreeze': {'values': [2, 4]},
         'model_name': {'values': ['ResNet', 'efficientnet', 'inception']},
         'lr': {'values': [0.00001]},
         'num_of_measurements': {'values': [512, 1024, 2048, 4096, 8192]},
         'reshape_size': {'values': ["64_128", "256_512", "1024_2048"]},
         }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Wrist_GI_Transfer_Learning')
wandb.agent(sweep_id, function=main)

# save the contents of accuracy_data dataframe to a file
accuracy_data.to_csv('accuracy_data_dif_nets_with_dif_shape_and_measurements.csv')
