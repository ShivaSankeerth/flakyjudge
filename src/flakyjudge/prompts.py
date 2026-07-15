"""Judge prompt templates.

The user-turn template is byte-identical to LMUnit's
(ContextualAI/LMUnit, lmunit/constants.py) so that results are directly
comparable to the paper. The template is FROZEN before any experiment runs;
see PREREGISTRATION.md.
"""

UNIT_TEST_PROMPT = (
    "Query: {query}\n\nResponse: {response}\n\nUnit Test: {natural_unit_test}"
)

# Off-the-shelf API judges (unlike LMUnit's trained checkpoints) need an
# explicit output-format instruction. Fixed across all judges and experiments.
JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a response satisfies a unit test criterion. "
    "Score how well the response satisfies the unit test on a scale of 1 to 5, "
    "where 1 means it clearly fails the criterion and 5 means it clearly "
    "satisfies it. Reply with a single digit from 1 to 5 and nothing else."
)

# E2b positional variant: same fields, mechanically reordered.
UNIT_TEST_PROMPT_REORDERED = (
    "Unit Test: {natural_unit_test}\n\nQuery: {query}\n\nResponse: {response}"
)


def build_prompt(query: str, response: str, unit_test: str, reordered: bool = False) -> str:
    template = UNIT_TEST_PROMPT_REORDERED if reordered else UNIT_TEST_PROMPT
    return template.format(query=query, response=response, natural_unit_test=unit_test)
