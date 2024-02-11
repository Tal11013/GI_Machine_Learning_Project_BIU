# Import the W&B Python Library and log into W&B
import wandb
from main_for_wrist import main_wrist

wandb.login()


def main():
    wandb.init()
    config = wandb.config
    for epoch in range(config.epoch):
        test_acc, test_loss = main_wrist(wandb.config)
        wandb.log({"Test accuracy": test_acc, "Test loss": test_loss, "Epoch": config.epoch})



# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'Test accuracy'},
    'parameters':
        {
            'lr': {'values': [0.0001]},
            'epoch': {'values': [10, 20, 50]},
            'shape': {'values': ["32_64"]},
            'num_of_measurements': {'values': [1024]},
            'batch_size': {'values': [16]},
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='GI_Machine_Learning_Project')
wandb.agent(sweep_id, function=main)
