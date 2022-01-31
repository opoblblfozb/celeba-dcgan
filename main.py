from configparser import ConfigParser

from train import get_trainer

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")

    config = config["default"]

    trainer = get_trainer(
        data_root=config.get("data_root"),
        ngpu=config.getint("ngpu"),
        lr=config.getfloat("lr"),
        beta1=config.getfloat("beta1"),
        num_epochs=config.getint("num_epochs"),
        image_size=config.getint("image_size"),
        batch_size=config.getint("batch_size"),
        workers=config.getint("workers"),
        nz=config.getint("nz"),
        nc=config.getint("nc"),
        ndf=config.getint("ndf"),
        ngf=config.getint("ngf"),
        pjname=config.get("pjname"),
        group=config.get("group"))
    trainer.train()
