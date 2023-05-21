"""
Implementation of the SwAV [1] method.

[1]. Mathilde Caron, et.al, Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav
"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.trainer_callbacks import ssl_trainer_callbacks
from pflbases.client_callbacks import local_completion_callbacks
from pflbases.models import swav

from pflbases import ssl_client
from pflbases import ssl_trainer
from pflbases import ssl_datasources



def main():
    """
    A pFL sesstion for SwaV approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=swav.SwaV,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
        ],
        trainer_callbacks=[
            separate_trainer_callbacks.PersonalizedModelMetricCallback,
            separate_trainer_callbacks.PersonalizedModelStatusCallback,
            ssl_trainer_callbacks.ModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        model=swav.SwaV,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
