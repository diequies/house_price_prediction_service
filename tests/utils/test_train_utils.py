""" Tests for the train utils methods """
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.utils.train_utils import get_inputs_from_signature


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
