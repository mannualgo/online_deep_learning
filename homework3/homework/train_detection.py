import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.return_types import mode
import torch.utils.tensorboard as tb


from .models import load_model, Classifier,save_model,ClassificationLoss,RegressionLoss

import sys
sys.path.insert(0, '/content/online_deep_learning/homework3/homework/datasets')

from road_dataset import load_data
from .metrics import DetectionMetric



def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 10,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = 2024,
   
    **kwargs,
):
    

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)
    
    
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    

    for datum in train_data:
        print(datum["image"].to(device).shape)
        print(datum["track"].to(device).shape)
        print(datum["depth"].to(device).shape)
        break
    

    # create loss function and optimizer & AccuracyMetric
    wt=torch.Tensor([0.2,0.9,0.9]).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=wt)
    loss_func_reg=RegressionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr) 
    trainAccuracyMetric= DetectionMetric()
    valAccuracyMetric= DetectionMetric()

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}
    print("batch size", batch_size)
    # training loop
   
    #image, depth, and track 
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        trainAccuracyMetric.reset()
        valAccuracyMetric.reset()
        
        for datum in train_data:
            img= datum["image"].to(device)
            track= datum["track"].to(device)
            depth=datum["depth"].to(device)
            
            
            # TODO: implement training step
            #preds: torch.Tensor,
            # labels: torch.Tensor,
            # depth_preds: torch.Tensor,
            # depth_labels: torch.Tensor,

            pred_lbl = model(img)

            
            loss_obtained_cls = loss_func(pred_lbl[0],track)
            loss_otained_reg=loss_func_reg(pred_lbl[1],depth)

            loss_obtained= loss_obtained_cls+loss_otained_reg

            prediction = model.predict(img)
            
            trainAccuracyMetric.add(prediction[0],track,prediction[1],depth)
            
            optimizer.zero_grad()
            loss_obtained.backward()
            optimizer.step()
  

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            
            for datum in val_data:
                img= datum["image"].to(device)
                track= datum["track"].to(device)
                depth=datum["depth"].to(device)
                prediction = model.predict(img)
                valAccuracyMetric.add(prediction[0],track,prediction[1],depth)

        computed_validation_metrics = valAccuracyMetric.compute()
        # log average train and val accuracy to tensorboard
        
        epoch_train_acc = trainAccuracyMetric.compute()["accuracy"]
        epoch_val_acc = computed_validation_metrics["accuracy"]

        epoch_val_acc = torch.as_tensor(computed_validation_metrics["accuracy"])
        epoch_iou = torch.as_tensor(computed_validation_metrics["iou"])
        epoch_abs_depth_error = torch.as_tensor(computed_validation_metrics["abs_depth_error"])
        epoch_tp_depth_error = torch.as_tensor(computed_validation_metrics["tp_depth_error"])

        
        logger.add_scalar("train_acc",epoch_train_acc,global_step)
        logger.add_scalar("val_acc",epoch_val_acc,global_step)
        #raise NotImplementedError("Logging not implemented")

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 2 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"accuracy={epoch_val_acc:.4f} "
                f"iou={epoch_iou:.4f} "
                f"abs_depth_error={epoch_abs_depth_error:.4f} "
                f"tp_depth_error={epoch_tp_depth_error:.4f} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))



