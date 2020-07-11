from src.trainer import Trainer
if __name__ == '__main__':
    trainer = Trainer("netmnist_00_0", r'.\cfg.ini')
    trainer.train()