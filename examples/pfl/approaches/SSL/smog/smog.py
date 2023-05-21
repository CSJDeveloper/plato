"""
Implementation of the SMoG [1] method.

[1]. Bo Pang, et.al, Unsupervised Visual Representation Learning by Synchronous Momentum Grouping.
ECCV, 2022. https://arxiv.org/pdf/2006.07733.pdf.

Source code: None
"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.trainer_callbacks import ssl_trainer_callbacks
from pflbases.client_callbacks import local_completion_callbacks
from pflbases.models import smog

from pflbases import ssl_client
from pflbases import ssl_datasources


from smog_trainer import Trainer


def main():
    """
    A personalized federated learning sesstion for SMoG approach.
    """
    trainer = Trainer
    client = ssl_client.Client(
        model=smog.SMoG,
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
        model=smog.SMoG,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
