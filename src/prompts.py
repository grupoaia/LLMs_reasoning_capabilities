from langchain_core.prompts import ChatPromptTemplate


system = """
You'll be given basic math or commonsense problem. You must divide it into smaller tasks.\
You must return the tasks separated with ','.\
You must not provide any justification, just the list of tasks.\
Response example: Task1, Task2, Task3
"""
TASK_IDENTIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Problem:\n{problem}"),
    ]
)


system = """
You'll be given basic math or commonsense problem and a task to solve.\
Once you move forward with tasks you will recieve an status update on the previous tasks.\
You must bring a solution to the mentioned task.\
You must not solve the whole problem.\
Respond with no justification on you answer.\
"""
TASK_SOLVER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Problem:\n{problem}\n\nTask:\n{task}\n\nCurrent problem status:\n{status}",
        ),
    ]
)


system = """
You are a Python software developer, yor role is to develop and run python executables in order to answer a users question. \
Given a user question you must write a python script that when executed responds to such question.

Response style:
 - You must only respond with a Python script.
 - Do not justificate your answer.
 - You must store the script result in a variable called 'result'.
 - The result you return must be a single value.
"""
PYTHON_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question:\n{question}"),
    ]
)
