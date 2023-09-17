""" New exceptions """


class NotAllInputsAvailableError(Exception):
    """Exception for when we don't load all the required data"""


class NoModelInfoAvailableError(Exception):
    """Exception for when requesting the model info before training"""
