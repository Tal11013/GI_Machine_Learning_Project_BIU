# Import the W&B Python Library and log into W&B
import wandb
from main_for_wrist import main_wrist

wandb.login(key = "046780ec306c02f19ce204adc33cbc86e04e80fd")


def main():
    wandb.init()
    #for epoch in range(config.epoch):
    test_acc, test_loss = main_wrist(wandb.config)
    print(test_acc, test_loss)


# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'Test accuracy'},
    'parameters':
        {
            'lr': {'values': [0.0002]},
            'epoch': {'values': [10, 25, 50]},
            'shape': {'values': ["64_128"]},
            'batch_size': {'values': [16]},
            'sampling_rate': {'values': [0.2, 0.35, 0.5]}
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='GI_Machine_Learning_Project')
wandb.agent(sweep_id, function=main)
