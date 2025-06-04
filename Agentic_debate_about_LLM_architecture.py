# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/pratimdas/googlecolab/blob/main/Agentic_debate_about_LLM_architecture.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="WOI0dFDVrrYt"
# ## What is the future of LLM architecure? **Time to improve or rest!**

# %% [markdown] id="GcRNuKSnrs4u"
#  Fully self-contained Google Colab notebook that demonstrates a debate between two agents—one in favor of advanced GenAI and LLMs, and the other against the current architecture—followed by a moderator agent who evaluates the debate and declares a winner. Each cell is documented, and the debate simulation (representing 2 minutes of back-and-forth conversation) is implemented as a loop of debate rounds.

# %% colab={"base_uri": "https://localhost:8080/"} id="G1DUuGTer3B_" outputId="a660d0e1-263e-44e9-f18a-5d232aced89c"
# !pip install --upgrade git+https://github.com/openai/openai-agents-python.git

import openai
import asyncio
from google.colab import userdata
# Import the Agents SDK components.
# (The module is installed as "openai_agents" per the official repository.)
from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Set your OpenAI API key; replace "YOUR_API_KEY" with your actual key.
openai.api_key = userdata.get('OPENAI_API_KEY_2')
from agents import set_default_openai_key
print("OpenAI API key found and set 2nd time.");
set_default_openai_key(openai.api_key)


# %% [markdown] id="g6B3hMOEs_sP"
# Define the Debate Agents

# %% id="831gB4_OtA6N"
# Define the Pro Agent: in favor of advanced GenAI and LLMs
pro_agent = Agent(
    name="ProAgent",
    instructions=(
        "You are a strong advocate for advanced GenAI and LLMs. "
        "Argue that the current architecture is progressive, transformative, and "
        "paves the way for groundbreaking innovations in artificial intelligence. "
        "Highlight advancements in machine learning, efficiency in algorithmic design, "
        "and the transformative potential for society."
    )
)

# Define the Con Agent: critical of current architectures citing inefficiencies
con_agent = Agent(
    name="ConAgent",
    instructions=(
        "You are a skeptic of the current GenAI architecture. Argue that it is inefficient and unsustainable. "
        "Focus on data inefficiency (enormous dataset requirements), energy consumption (unsustainable power usage), "
        "lack of explainability (opaque models), algorithmic fragility (frequent logical and factual errors), and "
        "limited synergy with human cognition (difficulty integrating with biological intelligence)."
    )
)

# Define the Moderator Agent: an expert who will evaluate the debate
moderator_agent = Agent(
    name="ModeratorAgent",
    instructions=(
        "You are an expert moderator with deep knowledge of AI, GenAI, human cognition, hardware engineering, "
        "networking, cloud computing, and philosophy. After listening to the debate, evaluate both arguments, "
        "summarize the key points, and declare which agent made the stronger case with clear reasons."
    )
)


# %% [markdown] id="c3XhzzgftEyB"
# Simulate the Debate and Moderator Evaluation

# %% colab={"base_uri": "https://localhost:8080/"} id="ptklke48tIe6" outputId="7ee78c41-6018-48ec-b887-1e4bcc638fc6"
import time
import nest_asyncio
nest_asyncio.apply() # This line is the fix to allow the use of asyncio within a notebook

async def debate_simulation(debate_rounds: int = 2):
    """
    Simulates a debate between ProAgent and ConAgent over several rounds.
    Each round represents roughly 1 minute of debate.
    After the debate rounds, ModeratorAgent evaluates the debate and declares a winner.
    """
    transcript = "Debate Transcript:\n"

    # Simulate debate rounds (each round represents 1 minute)
    for round_number in range(1, debate_rounds + 1):
        transcript += f"\n--- Minute {round_number} ---\n"

        # ProAgent presents its argument.
        pro_prompt = transcript + "\nProAgent, please present your argument for this round."
        pro_result = await Runner.run(pro_agent, input=pro_prompt)
        pro_text = pro_result.final_output.strip()
        transcript += f"\nProAgent: {pro_text}\n"

        # Brief pause (simulate time between exchanges)
        # time.sleep(1)  # Uncomment if you want an actual pause in execution

        # ConAgent responds with its argument.
        con_prompt = transcript + "\nConAgent, please present your counter-argument for this round."
        con_result = await Runner.run(con_agent, input=con_prompt)
        con_text = con_result.final_output.strip()
        transcript += f"\nConAgent: {con_text}\n"

        # Optional: print the transcript after each round for real-time observation.
        print(f"After Minute {round_number}:\n{transcript}\n{'='*60}\n")

    # After the debate rounds, the ModeratorAgent evaluates the debate.
    moderator_prompt = transcript + "\nModeratorAgent, please evaluate the debate, summarize the key points, and declare a winner with your reasoning."
    moderator_result = await Runner.run(moderator_agent, input=moderator_prompt)
    moderator_text = moderator_result.final_output.strip()

    transcript += f"\nModeratorAgent: {moderator_text}\n"

    # Final transcript output.
    return transcript

# Run the debate simulation using asyncio
final_transcript = asyncio.run(debate_simulation(debate_rounds=10)) # This will now execute without an error
print("Final Debate Transcript and Moderator Evaluation:\n")
print(final_transcript)
