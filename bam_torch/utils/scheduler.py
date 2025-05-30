import inspect
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup as warmup


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        config (dict): Optim dict from the input config
        optimizer (obj): torch optim object
    """

    def __init__(self, optimizer, optim_config, silent=False):
        self.optimizer = optimizer
        self.optim_config = optim_config.copy()
        self.warmup_scheduler = None
        self.silent = silent
        if self.optim_config.get("scheduler"):
            self.scheduler_type = self.optim_config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"

            #def scheduler_lambda_fn(x):
            #    return warmup_lr_lambda(x, self.optim_config)

            #self.optim_config["lr_lambda"] = scheduler_lambda_fn
            self.optim_config["lr_lambda"] = self.optim_config["decay_factor"]

        if (
            self.scheduler_type != "Null"
            and self.scheduler_type != "LinearWarmupCosineAnnealingLR"
        ):
            self.scheduler = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(self.optim_config)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)
            if self.scheduler_type == "ReduceLROnPlateau":
                if self.optim_config.get("decay_factor") == None:
                    self.optim_config["decay_factor"] = 0.9
                if self.optim_config.get("patience") == None:
                    self.optim_config["patience"] = 50
                if self.optim_config.get("threshold") == None:
                    self.optim_config["threshold"] = 1e-5
                if self.optim_config.get("cooldown") == None:
                    self.optim_config["cooldown"] = 0

                self.scheduler = lr_scheduler.ReduceLROnPlateau(
                                                optimizer, 
                                                factor=self.optim_config["decay_factor"], 
                                                patience=self.optim_config["patience"],
                                                threshold=self.optim_config["threshold"],
                                                cooldown=self.optim_config["cooldown"]

                                        )
                
        elif self.scheduler_type == "LinearWarmupCosineAnnealingLR":
            T_max = self.optim_config.get("fidelity_max_steps")
            if T_max is None:
                T_max = self.optim_config["max_steps"]
                if not self.silent:
                    print(f"Using max_steps for scheduler -> {T_max}")
            else:
                if not self.silent:
                    print(f"Using fidelity_max_steps for scheduler -> {T_max}")
            if self.optim_config["warmup_steps"] > 0:
                self.warmup_scheduler = warmup.ExponentialWarmup(
                    self.optimizer, warmup_period=self.optim_config["warmup_steps"],
                )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-7
            )
        

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            if self.warmup_scheduler:
                with self.warmup_scheduler.dampening():
                    self.scheduler.step(epoch)
            else:
                self.scheduler.step()

    def filter_kwargs(self, optim_config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: optim_config[arg] for arg in optim_config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]

    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, model_ckpt):
        return self.scheduler.load_state_dict(model_ckpt)

        
