import aletheia
from aletheia.signatures import Signature


class BasicGenerateInstruction(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will provide you with one or more ``example instruction(s)`` that were previously attempted for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        example_instructions = aletheia.InputField(desc="Example instruction(s) for this task.")
        proposed_instruction = aletheia.InputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithExamplesAndDataObservationsAndTip(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will give you a summary I have made about the dataset, along with some ``examples`` of the expected inputs and outputs for this task. I will also provide you with one or more ``example instruction(s)`` that were previously attempted for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        dataset_summary = aletheia.InputField(desc="Summary of the dataset.")
        examples = aletheia.InputField(desc="Example(s) of the task")
        example_instructions = aletheia.InputField(desc="Example instruction(s) for this task.")
        tip = aletheia.InputField(desc="A tip for something to keep in mind when generating the new instruction.")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithDataObservationsAndTip(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will give you a summary I have made about the dataset, along with some ``examples`` of the expected inputs and outputs for this task. I will also provide you with one or more ``example instruction(s)`` that were previously attempted for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        dataset_summary = aletheia.InputField(desc="Summary of the dataset.")
        example_instructions = aletheia.InputField(desc="Example instruction(s) for this task.")
        tip = aletheia.InputField(desc="A tip for something to keep in mind when generating the new instruction.")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithExamplesAndTip(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will give you some ``examples`` of the expected inputs and outputs for this task. I will also provide you with one or more ``example instruction(s)`` that were previously attempted for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        examples = aletheia.InputField(desc="Example(s) of the task")
        example_instructions = aletheia.InputField(desc="Example instruction(s) for this task.")
        tip = aletheia.InputField(desc="A tip for something to keep in mind when generating the new instruction.")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithTip(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will provide you with one or more ``example instruction(s)`` that were previously attempted for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        example_instructions = aletheia.InputField(desc="Example instruction(s) for this task.")
        tip = aletheia.InputField(desc="A tip for something to keep in mind when generating the new instruction.")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")


class BasicGenerateInstructionWithExamplesAndDataObservations(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will give you some ``observations`` I have made about the dataset and task, along with some ``examples`` of the expected inputs and outputs. I will also provide you with the current ``basic instruction`` that is being used for this task.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        observations = aletheia.InputField(desc="Observations about the dataset and task")
        examples = aletheia.InputField(desc="Example(s) of the task")
        basic_instruction = aletheia.InputField(desc="The initial instructions before optimization")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithExamples(aletheia.Signature):
        ("""You are an instruction optimizer for large language models. I will also give you one or more ``examples`` of the expected inputs and outputs.

Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        basic_instruction = aletheia.InputField(desc="The initial instructions before optimization")
        examples = aletheia.InputField(desc="Example(s) of the task")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")


class BasicGenerateInstructionWithDataObservations(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English.  I will also give you some ``observations`` I have made about the dataset and task. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = aletheia.InputField(desc="The initial instructions before optimization")
    observations = aletheia.InputField(desc="Observations about the dataset and task")
    proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")


class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = aletheia.InputField(desc="The initial instructions before optimization")
    proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionAllFields(Signature):
    """You are an instruction optimizer for large language models. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""
    ("""You are an instruction optimizer for large language models. I will provide you with""")
    ("""the current instruction that I am using, along with a template consisting of names""")
    ("""and prefixes for the different input and output fields for the model. Your task is to propose a new instruction""")
    ("""and template for these input and output fields that will help the language model perform the task well. Don't be afraid to be creative.""")

    initial_prompt_template = aletheia.InputField(desc="The initial prompt template for this task.")
    proposed_prompt_template = aletheia.OutputField(desc="The improved prompt template for this task.")
    # proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionOnly(Signature):
    ("""You are an instruction optimizer for large language models. I will provide you with an instruction I'm currently using. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative.""")

    current_instruction = aletheia.InputField(desc="The current instruction.")
    # examples_of_field_in_use = aletheia.InputField(desc="Examples of this field in use on examples from our task.")
    proposed_instruction = aletheia.OutputField(desc="The proposed instruction (reply with the instruction only).")

class BasicGenerateField(Signature):
    ("""You are an instruction optimizer for large language models. Your task is to propose a better string to use for one of the fields 
    in a prompt that is being inputted to a large language model to perform a certain task. The goal is for this improved field to improve 
    the performance of the language model on this task. Don't be afraid to be creative.""")

    current_field = aletheia.InputField(desc="The current string in use for this field.")
    examples_of_field_in_use = aletheia.InputField(desc="Examples of this field in use on examples from our task.")
    proposed_field = aletheia.OutputField(desc="The proposed string for the field (respond with the new field string only).")
    # proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(aletheia.Signature):
        """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with the corresponding validation scores, where higher scores indicate better quality. I will also include an example of each instruction in use on a randomly chosen sample from our validation set.

Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

        attempted_instructions = aletheia.InputField(format=dsp.passages2text)
        # attempted_instructions = aletheia.InputField(desc="Previously attempted task instructions, along with their resulting validation score, and an example of the instruction in use on a sample from our dataset.")
        proposed_instruction = aletheia.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = aletheia.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")