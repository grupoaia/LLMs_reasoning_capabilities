from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import SystemMessage, ToolMessage

from .graph_models import AgentState


class Agent:

    def __init__(self, tools, model="gpt-4o-mini"):
        self.system = """
            You are an adult expert in problem solving.\
            You will be given basic math and commonsense problems. Your task is to solve them.\
            To do so you can either write a Python script that solves the problem or divide the problem into smaller tasks and solve them \
            sequentially.
            Obviously, some users questions may be responded directly, it's on you to decide wether you respond directly or you call the tools. \
            You can only call the Python tool when specifically mentioned by the user, otherwise you should call the problem solving tool.
            You can call them multiple times but always sequentially.
            """
        self.tools = {t.name: t for t in tools}
        self.llm = self.llm = ChatOpenAI(model=model).bind_tools(tools)
        self.graph = None
        self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(":memory:")
        self.checkpointer = None

    async def init_graph(self):
        self.graph = await self.create_graph()

    async def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = await self.llm.ainvoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    async def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = await self.tools[t["name"]].ainvoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}

    async def create_graph(self):

        if self.checkpointer is None:
            self.checkpointer = await self.checkpointer_cm.__aenter__()

        bot = StateGraph(AgentState)

        bot.add_node("llm", self.call_openai)
        bot.add_node("action", self.take_action)

        bot.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )

        bot.add_edge("action", "llm")

        bot.set_entry_point("llm")

        graph = bot.compile(checkpointer=self.checkpointer)

        return graph

    async def close(self):
        # Properly close the checkpointer
        if self.checkpointer_cm:
            await self.checkpointer_cm.__aexit__(None, None, None)
            self.checkpointer = None
            self.checkpointer_cm = None
