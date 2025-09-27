import os


def init_swanlab_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    # if training_args.swanlab_entity is not None:
    #     os.environ["SWANLAB_ENTITY"] = training_args.swanlab_entity
    if training_args.wandb_project is not None:
        os.environ["SWANLAB_PROJECT"] = training_args.wandb_project
