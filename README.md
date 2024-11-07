# Setup

1. To setup the repo you must run `pip install -r requirements.txt` and you'll install all dependencies.
2. You must also set an .env with your you OpenAI api key as `OPENAI_API_KEY="whatsoever"`.

# What Does the repository offer

    1. You can access to the Benchmark notebook where you could find the process followed to obtain the results published in the Medium post: `notebooks/Benchmark.ipynb`
    2. You can run the app.py with `chainlit run app.py` which will run a chatbot interface in your `localhost:8000`.
    3. You can check how we built the Agent for the chatbot in the notebook located in: `notebooks/agent_creation.ipynb`

# The chatbot

# Run Chatbot

The goal of that is to allow you to see how easy is to build an Agent and what it offers. In this case the Agent is only developed to solve problems either via **Task decomposition** or via **Python script**. 

You can explore the Agent creation in `notebooks/agent_creation.ipynb`.