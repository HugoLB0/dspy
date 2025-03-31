# import inspect
# import logging
# import uuid
# from typing import Any

# # import aletheia.dsp as dsp
# import aletheia

# logger = logging.getLogger(__name__)
# #################### Assertion Helpers ####################


# def _build_error_msg(feedback_msgs):
#     """Build an error message from a list of feedback messages."""
#     return "\n".join([msg for msg in feedback_msgs])


# #################### Assertion Exceptions ####################


# class aletheiaAssertionError(AssertionError):
#     """Custom exception raised when a aletheia `Assert` fails."""

#     def __init__(
#         self,
#         id: str,
#         msg: str,
#         target_module: Any = None,
#         state: Any = None,
#         is_metric: bool = False,
#     ) -> None:
#         super().__init__(msg)
#         self.id = id
#         self.msg = msg
#         self.target_module = target_module
#         self.state = state
#         self.is_metric = is_metric


# class aletheiaSuggestionError(AssertionError):
#     """Custom exception raised when a aletheia `Suggest` fails."""

#     def __init__(
#         self,
#         id: str,
#         msg: str,
#         target_module: Any = None,
#         state: Any = None,
#         is_metric: bool = False,
#     ) -> None:
#         super().__init__(msg)
#         self.id = id
#         self.msg = msg
#         self.target_module = target_module
#         self.state = state
#         self.is_metric = is_metric


# #################### Assertion Primitives ####################


# class Constraint:
#     def __init__(
#         self,
#         result: bool,
#         msg: str = "",
#         target_module=None,
#         is_metric: bool = False,
#     ):
#         self.id = str(uuid.uuid4())
#         self.result = result
#         self.msg = msg
#         self.target_module = target_module
#         self.is_metric = is_metric

#         self.__call__()


# class Assert(Constraint):
#     """aletheia Assertion"""

#     def __call__(self) -> bool:
#         if isinstance(self.result, bool):
#             if self.result:
#                 return True
#             elif aletheia.settings.bypass_assert:
#                 logger.error(f"AssertionError: {self.msg}")
#                 return True
#             else:
#                 logger.error(f"AssertionError: {self.msg}")
#                 raise aletheiaAssertionError(
#                     id=self.id,
#                     msg=self.msg,
#                     target_module=self.target_module,
#                     state=aletheia.settings.trace,
#                     is_metric=self.is_metric,
#                 )
#         else:
#             raise ValueError("Assertion function should always return [bool]")


# class Suggest(Constraint):
#     """aletheia Suggestion"""

#     def __call__(self) -> Any:
#         if isinstance(self.result, bool):
#             if self.result:
#                 return True
#             elif aletheia.settings.bypass_suggest:
#                 logger.info(f"SuggestionFailed: {self.msg}")
#                 return True
#             else:
#                 logger.info(f"SuggestionFailed: {self.msg}")
#                 raise aletheiaSuggestionError(
#                     id=self.id,
#                     msg=self.msg,
#                     target_module=self.target_module,
#                     state=aletheia.settings.trace,
#                     is_metric=self.is_metric,
#                 )
#         else:
#             raise ValueError("Suggestion function should always return [bool]")


# #################### Assertion Handlers ####################


# def noop_handler(func):
#     """Handler to bypass assertions and suggestions.

#     Now both assertions and suggestions will become noops.
#     """

#     def wrapper(*args, **kwargs):
#         with aletheia.settings.context(bypass_assert=True, bypass_suggest=True):
#             return func(*args, **kwargs)

#     return wrapper


# def bypass_suggest_handler(func):
#     """Handler to bypass suggest only.

#     If a suggestion fails, it will be logged but not raised.
#     And If an assertion fails, it will be raised.
#     """

#     def wrapper(*args, **kwargs):
#         with aletheia.settings.context(bypass_suggest=True, bypass_assert=False):
#             return func(*args, **kwargs)

#     return wrapper


# def bypass_assert_handler(func):
#     """Handler to bypass assertion only.

#     If a assertion fails, it will be logged but not raised.
#     And If an assertion fails, it will be raised.
#     """

#     def wrapper(*args, **kwargs):
#         with aletheia.settings.context(bypass_assert=True):
#             return func(*args, **kwargs)

#     return wrapper


# def assert_no_except_handler(func):
#     """Handler to ignore assertion failure and return None."""

#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except aletheiaAssertionError:
#             return None

#     return wrapper


# def backtrack_handler(func, bypass_suggest=True, max_backtracks=2):
#     """Handler for backtracking suggestion and assertion.

#     Re-run the latest predictor up to `max_backtracks` times,
#     with updated signature if an assertion fails. updated signature adds a new
#     input field to the signature, which is the feedback.
#     """

#     def wrapper(*args, **kwargs):
#         error_msg, result = None, None
#         with aletheia.settings.lock:
#             aletheia.settings.backtrack_to = None
#             aletheia.settings.suggest_failures = 0
#             aletheia.settings.assert_failures = 0

#             # Predictor -> List[feedback_msg]
#             aletheia.settings.predictor_feedbacks = {}

#             current_error = None
#             for i in range(max_backtracks + 1):
#                 if i > 0 and aletheia.settings.backtrack_to is not None:
#                     # generate values for new fields
#                     feedback_msg = _build_error_msg(
#                         aletheia.settings.predictor_feedbacks[aletheia.settings.backtrack_to],
#                     )

#                     aletheia.settings.backtrack_to_args = {
#                         "feedback": feedback_msg,
#                         "past_outputs": past_outputs,
#                     }

#                 # if last backtrack: ignore suggestion errors
#                 if i == max_backtracks:
#                     if isinstance(current_error, aletheiaAssertionError):
#                         raise current_error
#                     aletheia.settings.trace.clear()
#                     result = bypass_suggest_handler(func)(*args, **kwargs) if bypass_suggest else None
#                     break
#                 else:
#                     try:
#                         aletheia.settings.trace.clear()
#                         result = func(*args, **kwargs)
#                         break
#                     except (aletheiaSuggestionError, aletheiaAssertionError) as e:
#                         if not current_error:
#                             current_error = e
#                         _error_id, error_msg, error_target_module, error_state = (
#                             e.id,
#                             e.msg,
#                             e.target_module,
#                             e.state[-1],
#                         )

#                         # increment failure count depending on type of error
#                         if isinstance(e, aletheiaSuggestionError) and e.is_metric:
#                             aletheia.settings.suggest_failures += 1
#                         elif isinstance(e, aletheiaAssertionError) and e.is_metric:
#                             aletheia.settings.assert_failures += 1

#                         if aletheia.settings.trace:
#                             if error_target_module:
#                                 for i in range(len(aletheia.settings.trace) - 1, -1, -1):
#                                     trace_element = aletheia.settings.trace[i]
#                                     mod = trace_element[0]
#                                     if mod == error_target_module:
#                                         error_state = e.state[i]
#                                         aletheia.settings.backtrack_to = mod
#                                         break
#                             else:
#                                 aletheia.settings.backtrack_to = aletheia.settings.trace[-1][0]

#                             if aletheia.settings.backtrack_to is None:
#                                 logger.error("Module not found in trace. If passing a aletheia Signature, please specify the intended module for the assertion (e.g., use `target_module = self.my_module(my_signature)` instead of `target_module =  my_signature`).")

#                             # save unique feedback message for predictor
#                             if error_msg not in aletheia.settings.predictor_feedbacks.setdefault(
#                                 aletheia.settings.backtrack_to,
#                                 [],
#                             ):
#                                 aletheia.settings.predictor_feedbacks[aletheia.settings.backtrack_to].append(error_msg)

#                             # use `new_signature` if available (CoT)
#                             if hasattr(error_state[0], 'new_signature'):
#                                 output_fields = error_state[0].new_signature.output_fields
#                             else:
#                                 output_fields = error_state[0].signature.output_fields
#                             past_outputs = {}
#                             for field_name in output_fields.keys():
#                                 past_outputs[field_name] = getattr(
#                                     error_state[2],
#                                     field_name,
#                                     None,
#                                 )

#                             # save latest failure trace for predictor per suggestion
#                             error_state[1]
#                             error_op = error_state[2].__dict__["_store"]
#                             error_op.pop("_assert_feedback", None)
#                             error_op.pop("_assert_traces", None)

#                         else:
#                             logger.error(
#                                 "UNREACHABLE: No trace available, this should not happen. Is this run time?",
#                             )

#             return result

#     return wrapper


# def handle_assert_forward(assertion_handler, **handler_args):
#     def forward(self, *args, **kwargs):
#         args_to_vals = inspect.getcallargs(self._forward, *args, **kwargs)

#         # if user has specified a bypass_assert flag, set it
#         if "bypass_assert" in args_to_vals:
#             aletheia.settings.configure(bypass_assert=args_to_vals["bypass_assert"])

#         wrapped_forward = assertion_handler(self._forward, **handler_args)
#         return wrapped_forward(*args, **kwargs)

#     return forward


# default_assertion_handler = backtrack_handler


# def assert_transform_module(
#     module,
#     assertion_handler=default_assertion_handler,
#     **handler_args,
# ):
#     """
#     Transform a module to handle assertions.
#     """
#     if not getattr(module, "forward", False):
#         raise ValueError(
#             "Module must have a forward method to have assertions handled.",
#         )
#     if getattr(module, "_forward", False):
#         logger.info(
#             f"Module {module.__class__.__name__} already has a _forward method. Skipping...",
#         )
#         pass  # TODO warning: might be overwriting a previous _forward method

#     module._forward = module.forward
#     module.forward = handle_assert_forward(assertion_handler, **handler_args).__get__(
#         module,
#     )

#     if all(
#         map(lambda p: isinstance(p[1], aletheia.retry.Retry), module.named_predictors()),
#     ):
#         pass  # we already applied the Retry mapping outside
#     elif all(
#         map(lambda p: not isinstance(p[1], aletheia.retry.Retry), module.named_predictors()),
#     ):
#         module.map_named_predictors(aletheia.retry.Retry)
#     else:
#         raise RuntimeError("Module has mixed predictors, can't apply Retry mapping.")

#     module._assert_transformed = True

#     return module
