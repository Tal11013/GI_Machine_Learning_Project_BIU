from main_for_wrist import main_wrist_dict

params = {
    "lr": {"values": [0.0002]},
    "epoch": {"values": [10]},
    "shape": {"values": ["128_256"]},
    "num_of_measurements": {"values": ["2048"]},
    "batch_size": {"values": ["16"]}
}


main_wrist_dict(params)