import aletheia
from aletheia.primitives.program import Module
from aletheia.signatures.field import OutputField
from aletheia.signatures.signature import ensure_signature, Signature
from pydantic.fields import FieldInfo
from typing import Optional, Union, Type


class ChainOfThought(Module):
    
    def __init__(
        self, 
        signature: Type[Signature], 
        rationale_field: Optional[Union[OutputField, FieldInfo]] = None, 
        rationale_field_type: Type = str,
        **config
    ):
        """
        A module that reasons step by step in order to predict the output of a task.
        
        Args:
            signature (Type[aletheia.Signature]): The signature of the module.
            rationale_field (Optional[Union[aletheia.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
            rationale_field_type (Type): The type of the rationale field.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else aletheia.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = aletheia.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
