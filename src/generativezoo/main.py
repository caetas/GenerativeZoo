# -*- coding: utf-8 -*-

"""Main module."""
from logger import FhpLogger

fhplog = FhpLogger(
    config_file_path="~/zuliprc-old",
    user_id="tomas.pereira@aicos.fraunhofer.pt",
    to=["Logging"],
    msg_type="stream",
    topic="sc4c",
)


@fhplog.train_logger
def train():
    for i in range(20):
        if i % 10 == 0:
            fhplog.send_message("avg loss: %.5f, epoch: %d" % (0, i))


def main():
    """

    :return:
    """
    pass


if __name__ == "__main__":
    main()
