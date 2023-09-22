""" Tests for the train utils methods """
import pytest
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.utils.exceptions import EmptyModelSignatureError
from src.utils.train_utils import get_inputs_from_signature


def test_load_inputs_signature_empty():
    """
    Test that the method raises a exception when the signature is empty
    """
    input_schema = Schema(inputs=[])
    signature = ModelSignature(inputs=input_schema)

    with pytest.raises(EmptyModelSignatureError):
        get_inputs_from_signature(signature=signature)


def test_load_right_inputs_schema():
    """
    Test that extracts the right input names
    """
    input_schema = Schema(
        inputs=[ColSpec("float", "floor_m_sqrt"), ColSpec("integer", "garden")]
    )
    signature = ModelSignature(inputs=input_schema)

    expected_inputs = ["floor_m_sqrt", "garden"]

    assert get_inputs_from_signature(signature=signature) == expected_inputs
