import hydra
import torch
import wandb
import os
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from utils.trainer import (
    train_n_iters,
    test_n_iters,
    get_benign_metrics,
    get_advers_metrics,
    get_validation_metrics,
    visualize1,
    visualize2,
    validate,
    test_fid,
)
from torch.amp.grad_scaler import GradScaler
from datetime import datetime
from itertools import chain


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # Create an experimental folder using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = f"experiments/{timestamp}"
    os.makedirs(exp_folder, exist_ok=True)

    OmegaConf.resolve(cfg)
    wandb.init(project="Neural-Invisiblity-Cloak", config=OmegaConf.to_container(cfg))

    model_ckpt = None
    optimizer_ckpt = None
    critic_ckpt = None
    optimizer_critic_ckpt = None
    if cfg.continue_ckpt is not None:
        print(f"Detect Ckpt: {cfg.continue_ckpt}")
        all_ckpts: dict = torch.load(cfg.continue_ckpt, weights_only=True, map_location="cpu")
        model_ckpt = all_ckpts.get("model", None)
        optimizer_ckpt = all_ckpts.get("optimizer", None)
        critic_ckpt = all_ckpts.get("critic", None)
        optimizer_critic_ckpt = all_ckpts.get("optimizer_critic", None)
    elif (not cfg.models.from_scratch) and (cfg.models.pretrained is not None):
        model_ckpt = torch.load(cfg.models.pretrained, weights_only=True, map_location="cpu")

    if hasattr(cfg.attacks, "backdoor1"):
        backdoor1 = instantiate(cfg.attacks.backdoor1)
        backdoor2 = instantiate(cfg.attacks.backdoor2)
    else:
        backdoor1 = backdoor2 = instantiate(cfg.attacks)

    trainloader = instantiate(cfg.models.datasets.trainloader)
    testloader = instantiate(cfg.models.datasets.testloader)
    if hasattr(cfg.models.datasets, "valloader"):
        valloader = instantiate(cfg.models.datasets.valloader)
    else:
        valloader = None

    model = instantiate(cfg.models.modules)
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt)
    model.cuda()

    if cfg.models.distill:
        distill_model = instantiate(cfg.models.modules)
        distill_ckpt = torch.load(cfg.models.distill_ckpt, weights_only=True, map_location="cpu")
        if "model" in distill_ckpt:
            distill_ckpt = distill_ckpt["model"]
        distill_model.load_state_dict(distill_ckpt)
        distill_model.cuda().eval()
        for p in distill_model.parameters():
            p.requires_grad_(False)
    else:
        distill_model = None

    parameters = model.parameters()
    optimizer_fn = instantiate(cfg.models.optimizer)
    optimizer = optimizer_fn(parameters)
    if optimizer_ckpt is not None:
        optimizer.load_state_dict(optimizer_ckpt)

    scaler = GradScaler() if cfg.models.use_amp else None

    pixel_loss = instantiate(cfg.models.losses.l1loss)
    perceptual_loss = instantiate(cfg.models.losses.percloss).cuda() if cfg.models.use_vgg else None
    if hasattr(backdoor1, "poisoning_rate"):
        poisoning_only = True
    else:
        poisoning_only = False

    if cfg.models.use_gan:
        critic = instantiate(cfg.models.losses.ganloss).cuda()
        optimizer_critic = optimizer_fn(critic.parameters())
        scaler_critic = GradScaler() if cfg.models.use_amp else None
        if critic_ckpt is not None:
            critic.load_state_dict(critic_ckpt)
        if optimizer_critic_ckpt is not None:
            optimizer_critic.load_state_dict(optimizer_critic_ckpt)
    else:
        critic = None
        optimizer_critic = None
        scaler_critic = None

    benign_metrics = get_benign_metrics()
    advers_metrics = get_advers_metrics()
    validation_metrics = get_validation_metrics()
    n_iter_fid = 10000 // cfg.models.datasets.testloader.batch_size

    for e in range(1, cfg.epoch + 1):
        train_n_iters(
            trainloader,
            model,
            pixel_loss,
            optimizer,
            cfg.n_iter_per_epoch,
            scaler=scaler,
            backdoor=backdoor1,
            critic=critic,
            optimizer_critic=optimizer_critic,
            scaler_critic=scaler_critic,
            perceptual_loss=perceptual_loss,
            l1loss_weight=cfg.models.losses.l1loss_weight,
            percloss_weight=cfg.models.losses.percloss_weight,
            ganloss_weight=cfg.models.losses.ganloss_weight,
            gp_lambda=cfg.models.losses.gp_lambda,
            poisoning_only=poisoning_only,
            distill_model=distill_model,
        )
        res1, res2, res3, res4, res5, res6 = {}, {}, {}, {}, {}, {}
        res1 = test_n_iters(testloader, model, benign_metrics, cfg.n_iter_test, backdoor=None)
        res2 = test_n_iters(testloader, model, advers_metrics, cfg.n_iter_test, backdoor=backdoor2)
        if valloader is not None:
            res3 = validate(valloader, model, cfg.n_iter_val, validation_metrics)
        if e % cfg.itv_fid == 0:
            res4 = test_fid(testloader, model, n_iter_fid, backdoor2)
        if e % cfg.itv_vis == 0:
            x, y, z = visualize1(testloader, model, backdoor2)
            res5 = {
                "test/input": wandb.Image(x),
                "test/target": wandb.Image(y),
                "test/output": wandb.Image(z),
            }
            if valloader is not None:
                x, y, z = visualize2(valloader, model)
                res6 = {
                    "val/input": wandb.Image(x),
                    "val/target": wandb.Image(y),
                    "val/output": wandb.Image(z),
                }
        result = {**res1, **res2, **res3, **res4, **res5, **res6}
        wandb.log(result, step=e * cfg.n_iter_per_epoch)

        # Save model checkpoint at each epoch
        if e % cfg.itv_ckpt == 0:
            checkpoint_path = f"{exp_folder}/model_epoch_{e}.pth"
            checkpoint = {
                "epoch": e,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if critic is not None:
                checkpoint["critic"] = critic.state_dict()
                checkpoint["optimizer_critic"] = optimizer_critic.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
