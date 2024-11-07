from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .utils import PythonScriptParser
from .prompts import PYTHON_GENERATOR_PROMPT, TASK_IDENTIFIER_PROMPT, TASK_SOLVER_PROMPT


def problem_solver_sync(question, show=False):
    """Tool that solves problems dividing them into smaller ones."""

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = TASK_IDENTIFIER_PROMPT | llm
    response = chain.invoke({"problem": question}).content
    tasks = [r.strip() for r in response.split(",")]

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = TASK_SOLVER_PROMPT | llm

    status = ""
    for i, task in enumerate(tasks):
        response = chain.invoke(
            {"problem": question, "task": task, "status": status}
        ).content
        status += f"{task}: {response}\n"
        if show:
            print(f"Task {i + 1}:\n{task}-{response}")
            print("-" * 50)
    return status


@tool
async def problem_solver(question, show=False):
    """Tool that solves problems dividing them into smaller ones."""

    status = problem_solver_sync(question, show=show)

    return status


def python_script_sync(question):
    """Tool to generate python script if the question is mathematics related."""

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = PYTHON_GENERATOR_PROMPT | llm | PythonScriptParser()
    response = chain.invoke({"question": question})

    variables = {}
    try:
        # pylint: disable-next=exec-used
        exec(response, variables)
    except Exception as e:
        print(response)
        raise e
    return response, variables["result"]


@tool
async def python_script(question):
    """Tool to generate python script if the question is mathematics related."""

    response, result = python_script_sync(question)

    python_response = (
        f"Python script:\n{response}\n\n Result from executing Python script:\n{result}"
    )
    return python_response
