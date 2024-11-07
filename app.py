import os
from io import BytesIO
import chainlit as cl
from openai import AsyncOpenAI
from chainlit.element import ElementBased
from langchain_core.messages import HumanMessage

from src.agent import Agent
from src.tools import python_script, problem_solver


@cl.on_chat_start
async def on_chat_start():

    tools = [python_script, problem_solver]
    agent = Agent(tools)
    await agent.init_graph()

    cl.user_session.set("runnable", agent)

    await cl.Message(
        content="Hey! It's Grupo AIA assistant here! What can I do for you today?",
        author="Grupo AIA",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("runnable")
    user_id = cl.user_session.get("id")

    inputs = {"messages": [HumanMessage(content=message.content)]}
    res = await app.graph.ainvoke(
        inputs,
        config={
            "configurable": {"thread_id": user_id, "recursion_limit": 5},
            "callbacks": [cl.LangchainCallbackHandler()],
        },
    )
    print("response ==>", res)
    await cl.Message(content=res["messages"][-1].content).send()


cl.instrument_openai()
client = AsyncOpenAI()


@cl.step(type="tool")
async def speech_to_text(audio_file):

    print(type(audio_file))
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@cl.step(type="tool")
async def generate_text_from_transcription(message):

    runnable = cl.user_session.get("runnable")
    user_id = cl.user_session.get("id")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"input": message},
        config={
            "configurable": {"session_id": user_id},
            "callbacks": [cl.LangchainCallbackHandler()],
        },
    ):
        await msg.stream_token(chunk)


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):

    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):

    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)

    transcription = await speech_to_text(whisper_input)

    await generate_text_from_transcription(transcription)


@cl.on_chat_end
async def on_chat_end():
    agent = cl.user_session.get("agent")
    if agent:
        await agent.close()
