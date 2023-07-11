from hparms import Training_Hparams
from Trainer import Trainer
from utils import same_seeds

if __name__ == "__main__":

    same_seeds(2022)

    hp = Training_Hparams(feature_set="feature_Fbank", n_features=40, model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Fbank_and_Phonation", n_features=50, model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec", n_features=40, model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec_and_Phonation", n_features=50, model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC",n_features=40,model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC_and_Phonation",n_features=50,model="CNN")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    same_seeds(2023)

    hp = Training_Hparams(feature_set="feature_Fbank", n_features=40, model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Fbank_and_Phonation", n_features=50, model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec", n_features=40, model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec_and_Phonation", n_features=50, model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC",n_features=40,model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC_and_Phonation",n_features=50,model="CNN_LSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    same_seeds(2024)

    hp = Training_Hparams(feature_set="feature_Fbank", n_features=40, model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Fbank_and_Phonation", n_features=50, model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec", n_features=40, model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_Melspec_and_Phonation", n_features=50, model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC",n_features=40,model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()

    hp = Training_Hparams(feature_set="feature_MFCC_and_Phonation",n_features=50,model="CNN_BiLSTM")
    trainer = Trainer(hp)
    trainer.set_trainer_configuration()
    trainer.start_trainer()