import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

# from tutils import open_dev_mode
import swanlab
from swanlab.integration.accelerate import SwanLabTracker

# swanlab.login(open_dev_mode())

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger
import time
import fire


def main(exp="1gpu"):
    # hyperparameters
    config = {
        "num_epoch": 5,
        "batch_num": 64,
        "learning_rate": 1e-3,
        "report_step_num": 20,
    }

    # Download the raw CIFAR-10 data.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    BATCH_SIZE = config["batch_num"]
    my_training_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    my_testing_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    # Using resnet18 model, make simple changes to fit the data set
    my_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    my_model.conv1 = torch.nn.Conv2d(
        my_model.conv1.in_channels, my_model.conv1.out_channels, 3, 1, 1
    )
    my_model.maxpool = torch.nn.Identity()
    my_model.fc = torch.nn.Linear(my_model.fc.in_features, 10)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    my_optimizer = torch.optim.SGD(
        my_model.parameters(), lr=config["learning_rate"], momentum=0.9
    )

    # Init accelerate with swanlab tracker
    tracker = SwanLabTracker("SPEED_WITH_DDP", experiment_name=exp)
    accelerator = Accelerator(log_with=tracker)
    accelerator.init_trackers("SPEED_WITH_DDP", config=config)
    my_model, my_optimizer, my_training_dataloader, my_testing_dataloader = (
        accelerator.prepare(
            my_model, my_optimizer, my_training_dataloader, my_testing_dataloader
        )
    )
    device = accelerator.device
    my_model.to(device)

    # Get logger
    logger = get_logger(__name__)

    # Begin training
    start_train_time = time.time()
    stp_time = time.time()
    for ep in range(config["num_epoch"]):
        epoch_time = time.time()
        # train model
        if accelerator.is_local_main_process:
            print(f"begin epoch {ep} training...")
        step = 0
        for stp, data in enumerate(my_training_dataloader):
            my_optimizer.zero_grad()
            inputs, targets = data
            outputs = my_model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            my_optimizer.step()
            if config["report_step_num"] > 0 and stp % config["report_step_num"] == 0:
                stp_end_time = time.time()
                accelerator.log(
                    {
                        "training loss": loss,
                        "epoch num": ep,
                        "used time": time.time() - start_train_time,
                        "step time": (stp_end_time - stp_time)
                        / config["report_step_num"],
                    },
                    step=ep * len(my_training_dataloader) + stp,
                )
                stp_time = stp_end_time
            if accelerator.is_local_main_process:
                print(
                    f"train epoch {ep} [{stp}/{len(my_training_dataloader)}] | train loss {loss}"
                )
        accelerator.log(
            {
                "train epoch time": time.time() - epoch_time,
            },
        )

        # eval model
        if accelerator.is_local_main_process:
            print(f"begin epoch {ep} evaluating...")
        total_acc_num = 0
        start_eval_time = time.time()
        for stp, (inputs, targets) in enumerate(my_testing_dataloader):
            predictions = my_model(inputs)
            predictions = torch.argmax(predictions, dim=-1)
            # Gather all predictions and targets
            all_predictions, all_targets = accelerator.gather_for_metrics(
                (predictions, targets)
            )
            acc_num = (all_predictions.long() == all_targets.long()).sum()
            total_acc_num += acc_num
            if accelerator.is_local_main_process:
                print(
                    f"eval epoch {ep} [{stp}/{len(my_testing_dataloader)}] | eval acc {acc_num/len(all_targets)}"
                )
        eval_time = time.time() - start_eval_time
        if accelerator.is_local_main_process:
            print(
                f"eval acc {total_acc_num / len(my_testing_dataloader.dataset)} | use time: {eval_time}"
            )
        accelerator.log(
            {
                "eval acc": total_acc_num / len(my_testing_dataloader.dataset),
                "eval time": eval_time,
            }
        )

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print(f"FINISH TRAINING")
        print(f"TOTAL USED {time.time()-start_train_time}s")
        print(f"SAVING MODEL...")
    accelerator.save_model(my_model, "outputs")

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
