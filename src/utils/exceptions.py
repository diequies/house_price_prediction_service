""" New exceptions """


class NotAllInputsAvailableError(Exception):
    """Exception for when we don't load all the required data"""


class NoModelInfoAvailableError(Exception):
    """Exception for when requesting the model info before training"""


class MissingColumnError(Exception):
    """Exception for when we are missing a required column"""
