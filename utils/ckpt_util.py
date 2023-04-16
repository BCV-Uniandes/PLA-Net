import os
import copy
import torch
import shutil
from collections import OrderedDict
import logging
import numpy as np


def save_ckpt(
    model,
    optimizer,
    train_epoch_loss,
    val_epoch_loss,
    train_epoch_nap,
    val_epoch_nap,
    epoch,
    save_path,
    name_pre,
    name_post="best",
):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
        "epoch": epoch,
        "model_state_dict": model_cpu,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_epoch_loss,
        "val_loss": val_epoch_loss,
        "train_map": train_epoch_nap,
        "val_map": val_epoch_nap,
    }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = "{}/{}_{}.pth".format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print("model has been saved as {}".format(filename))


def load_pretrained_models(
    model, pretrained_model, phase, ismax=True
):  # ismax means max best
    if ismax:
        best_value = -np.inf
    else:
        best_value = np.inf
    epoch = -1

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            logging.info("===> Loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            try:
                best_value = checkpoint["best_value"]
                if best_value == -np.inf or best_value == np.inf:
                    show_best_value = False
                else:
                    show_best_value = True
            except:
                best_value = best_value
                show_best_value = False

            model_dict = model.state_dict()
            ckpt_model_state_dict = checkpoint["state_dict"]

            # rename ckpt (avoid name is not same because of multi-gpus)
            is_model_multi_gpus = True if list(model_dict)[0][0][0] == "m" else False
            is_ckpt_multi_gpus = (
                True if list(ckpt_model_state_dict)[0][0] == "m" else False
            )

            if not (is_model_multi_gpus == is_ckpt_multi_gpus):
                temp_dict = OrderedDict()
                for k, v in ckpt_model_state_dict.items():
                    if is_ckpt_multi_gpus:
                        name = k[7:]  # remove 'module.'
                    else:
                        name = "module." + k  # add 'module'
                    temp_dict[name] = v
                # load params
                ckpt_model_state_dict = temp_dict

            model_dict.update(ckpt_model_state_dict)
            model.load_state_dict(ckpt_model_state_dict)

            if show_best_value:
                logging.info(
                    "The pretrained_model is at checkpoint {}. \t "
                    "Best value: {}".format(checkpoint["epoch"], best_value)
                )
            else:
                logging.info(
                    "The pretrained_model is at checkpoint {}.".format(
                        checkpoint["epoch"]
                    )
                )

            if phase == "train":
                epoch = checkpoint["epoch"]
            else:
                epoch = -1
        else:
            raise ImportError(
                "===> No checkpoint found at '{}'".format(pretrained_model)
            )
    else:
        logging.info("===> No pre-trained model")
    return model, best_value, epoch


def load_pretrained_optimizer(
    pretrained_model, optimizer, scheduler, lr, use_ckpt_lr=True
):
    if pretrained_model:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            if "optimizer_state_dict" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if "scheduler_state_dict" in checkpoint.keys():
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if use_ckpt_lr:
                    try:
                        lr = scheduler.get_lr()[0]
                    except:
                        lr = lr

    return optimizer, scheduler, lr


def save_checkpoint(state, is_best, save_path, postname):
    filename = "{}/{}_{}.pth".format(save_path, postname, int(state["epoch"]))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}/{}_best.pth".format(save_path, postname))


def change_ckpt_dict(model, optimizer, scheduler, opt):
    for _ in range(opt.epoch):
        scheduler.step()
    is_best = opt.test_value < opt.best_value
    opt.best_value = min(opt.test_value, opt.best_value)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    # optim_cpu = {k: v.cpu() for k, v in optimizer.state_dict().items()}
    save_checkpoint(
        {
            "epoch": opt.epoch,
            "state_dict": model_cpu,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_value": opt.best_value,
        },
        is_best,
        opt.save_path,
        opt.post,
    )


def load_models(model, device):
    print("------Copying model 1---------")
    prop_predictor1 = copy.deepcopy(model)

    print("------Copying model 2---------")
    prop_predictor2 = copy.deepcopy(model)

    print("------Copying model 3---------")
    prop_predictor3 = copy.deepcopy(model)

    print("------Copying model 4---------")
    prop_predictor4 = copy.deepcopy(model)

    test_model_path = "/media/SSD0/cigonzalez/drugs-discovery/BINARY_" + target

    test_model_path1 = test_model_path + "/Fold1/Best_Model.pth"
    test_model_path2 = test_model_path + "/Fold2/Best_Model.pth"
    test_model_path3 = test_model_path + "/Fold3/Best_Model.pth"
    test_model_path4 = test_model_path + "/Fold4/Best_Model.pth"

    # LOAD MODELS
    print("------- Loading weights----------")
    ckpt1 = torch.load(test_model_path1, map_location=lambda storage, loc: storage)

    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ].t()
    ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ] = ckpt1["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ].t()

    prop_predictor1.load_state_dict(ckpt1["model_state_dict"])
    prop_predictor1.to(device)

    ckpt2 = torch.load(test_model_path2, map_location=lambda storage, loc: storage)

    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ].t()
    ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ] = ckpt2["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ].t()

    prop_predictor2.load_state_dict(ckpt2["model_state_dict"])
    prop_predictor2.to(device)

    ckpt3 = torch.load(test_model_path3, map_location=lambda storage, loc: storage)
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ].t()
    ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ] = ckpt3["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ].t()

    prop_predictor3.load_state_dict(ckpt3["model_state_dict"])
    prop_predictor3.to(device)

    ckpt4 = torch.load(test_model_path4, map_location=lambda storage, loc: storage)

    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.0.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.1.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.2.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.3.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.4.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.5.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.6.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.7.weight"
    ].t()
    ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ] = ckpt4["model_state_dict"][
        "molecule_gcn.atom_encoder.atom_embedding_list.8.weight"
    ].t()

    prop_predictor4.load_state_dict(ckpt4["model_state_dict"])
    prop_predictor4.to(device)

    return prop_predictor1, prop_predictor2, prop_predictor3, prop_predictor4
