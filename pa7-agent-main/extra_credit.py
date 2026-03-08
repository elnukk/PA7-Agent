"""
Extra Credit 

In this assignment, you will create your own AI agent using DSPy and any tools/APIs you choose.

Your agent should:
1. Use DSPy's framework (signatures, modules, etc.)
2. Implement some interesting functionality beyond basic chat
3. Demonstrate your implementation with a working example

You have complete freedom to:
- Choose what your agent does (data analysis, creative generation, etc.)
- Select which APIs or libraries to integrate
- Design the interface and interaction pattern

Some ideas to get you started:
- A research assistant that uses web search
- A creative writing collaborator with style adaptation
- A data analysis agent that processes files
- A multi-step reasoning agent for complex problems
- Something completely different! Be creative!

Complete the TODOs below to build your agent.
"""

import dspy
import os
from api_keys import TOGETHER_API_KEY


# Configure environment
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# TODO: Add any imports you need for your agent
# Examples:
# from mem0 import Memory
# import requests
# import pandas as pd
import json
import re
from mem0 import Memory
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich import box

from agent import WebTools, memory_config

# TODO: Add any configuration your agent needs
# Examples:
# - API configurations
# - Model settings
# - Tool configurations
# Memory configuration

memory_config = {
    "llm": {
        "provider": "together",
        "config": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "together",
        "config": {
            "model": "intfloat/multilingual-e5-large-instruct"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1024
        }
    }
}

memory = Memory.from_config(memory_config)
console = Console()



# UI State
ui_state = {
    "mood": "Unknown",
    "mood_emoji": "🎬",
    "mood_desc": "Tell me how you feel!",
}

# TODO: Implement any helper classes or functions your agent needs
def detect_mood(user_input: str) -> str:
    lm = dspy.LM("together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct")
    prompt = f"""Analyze the user's message and detect their mood. Return ONLY a JSON object with these exact fields:
    - "mood": single word (Happy, Sad, Excited, Anxious, Relaxed, Adventurous, Romantic, Bored, Angry, Nostalgic)
    - "emoji": single emoji for the mood
    - "genres": list of 2-3 genres matching the mood (Action, Thriller, Comedy, Drama, Sci-Fi, Romance, Horror, Adventure)
    - "description": 5-7 word phrase describing what kind of movie would suit them.
    User message: "{user_input}"
    Return ONLY the JSON, NO EXTRA TEXT."""
    response = lm(messages=[{"role": "user", "content": prompt}])
    raw = response[0] if isinstance(response, list) else response
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            # update sidebar state
            ui_state["mood"] = data.get("mood", "Neutral")
            ui_state["mood_emoji"] = data.get("emoji", "🎬")
            ui_state["mood_desc"] = data.get("description", "")
            return json.dumps(data)
        except json.JSONDecodeError:
            pass

# TODO: Define DSPy Signature for your agent
# This specifies what your agent takes as input and produces as output
# Include instructions in the docstring about how it should behave
class YourAgentSignature(dspy.Signature):
    """
    You are a mood-aware movie assistant. You help users discover movies based on
    how they feel.

    Tool usage rules:
    • detect_mood: ALWAYS call FIRST to detect emotional state from the user's message.
    • web_search: after detecting mood, call with a query like 
      "best [genre1] [genre2] movies for [mood] mood" to find real recommendations.
      Always use web search for movie recommendations, never make them up.

    Be warm, conversational, and reference the user's mood naturally in your response.
    Always explain WHY a movie matches the mood based on the web search results.
    """
    user_input: str = dspy.InputField(desc="The user's message")  # TODO: Modify as needed
    response: str = dspy.OutputField(desc="Mood-aware response with recommendations")   # TODO: Modify as needed

# TODO: Implement your agent as a DSPy Module
class YourAgent(dspy.Module):
    """Mood-aware movie agent with web search recommendations"""

    def __init__(self):
        super().__init__()
        self.web_tools = WebTools()

        self.tools = [
            detect_mood,
            self.web_tools.web_search,
        ]
        self.agent = dspy.ReAct(
            signature=YourAgentSignature,
            tools=self.tools,
            max_iters=8,
        )

        # TODO: Initialize any components your agent needs
        # Examples:
        # - Tools for ReAct
        # - Memory systems
        # - API clients
        # - Other modules
        
        # TODO: Choose and initialize your DSPy module
        # Options:
        # - dspy.ReAct for tool-using agents
        # - dspy.ChainOfThought for reasoning
        # - dspy.Predict for simple prediction
        
        # Example for ReAct:
        # self.tools = [...]
        # self.agent = dspy.ReAct(
        #     signature=YourAgentSignature,
        #     tools=self.tools,
        #     max_iters=6
        # )

    def forward(self, user_input: str):
        """Process user input and generate a response."""
        # TODO: Implement your agent's logic
        # This should call your DSPy module and return results
        return self.agent(user_input=user_input)


# UI Helpers so it looks PRETTY
def build_sidebar() -> Panel:
    """Build the right-side panel showing mood."""
    content = Table.grid(padding=(0, 1))
    content.add_column(style="bold")
    content.add_column()

    content.add_row(Text("MOOD", style="bold cyan"), Text(""))
    content.add_row(
        Text(f"{ui_state['mood_emoji']}", style="bold"),
        Text(ui_state["mood"], style="bright_white bold"),
    )
    content.add_row("", Text(ui_state["mood_desc"], style="italic dim"))
    content.add_row("", "")

    return Panel(
        content,
        title="[bold magenta]🎬 Status[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    )


def build_trajectory_panel(trajectory: dict) -> Panel:
    """Format the ReAct trajectory cleanly."""
    lines = []
    i = 0
    while f"thought_{i}" in trajectory:
        thought = trajectory[f"thought_{i}"].strip().lstrip("]").strip()
        tool = trajectory.get(f"tool_name_{i}", "")
        args = trajectory.get(f"tool_args_{i}", {})
        obs = trajectory.get(f"observation_{i}", "")

        lines.append(f"[bold yellow]💭 Thought:[/bold yellow] {thought}")
        if tool and tool != "finish":
            lines.append(f"[bold blue]🔧 Tool:[/bold blue] [cyan]{tool}[/cyan]  [dim]{args}[/dim]")
            lines.append(f"[bold green]👁  Result:[/bold green] {str(obs)[:300]}")
        lines.append("")
        i += 1

    text = "\n".join(lines) if lines else "[dim]No steps.[/dim]"
    return Panel(
        text,
        title="[bold yellow]Chain of Thought[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )


def print_turn(prediction):
    """Render a full turn: trajectory + response + sidebar."""
    layout = Layout()
    layout.split_row(
        Layout(name="main", ratio=3),
        Layout(name="sidebar", ratio=1),
    )

    main_content = Layout()
    main_content.split_column(
        Layout(build_trajectory_panel(prediction.trajectory), name="trajectory", ratio=3),
        Layout(
            Panel(
                Text(prediction.response, style="bright_white"),
                title="[bold green]🤖 Agent Response[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            ),
            name="response",
            ratio=1,
        ),
    )

    layout["main"].update(main_content)
    layout["sidebar"].update(build_sidebar())
    console.print(layout)




def run_demo():
    """Demonstration of your agent."""
    
    # Configure DSPy
    lm = dspy.LM(model='together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1')
    dspy.configure(lm=lm)

    # TODO: Initialize your agent
    agent = YourAgent()

    console.print(Panel(
        "[bold magenta]🎬 Mood-Aware Movie Agent — Demo[/bold magenta]\n",
        border_style="magenta",
        box=box.DOUBLE,
    ))



    # TODO: Create test cases that demonstrate your agent's capabilities
    # Show what makes your agent interesting and useful!
    
    test_inputs = [
        "I'm feeling really adventurous, what should I watch?",
        "I'm a bit sad and nostalgic, recommend something",
    ]

    # TODO: Run your demo
    # for user_input in test_inputs:
    #     print(f"\n📝 User: {user_input}")
    #     response = agent(user_input=user_input)
    #     print(f"🤖 Agent: {response}")

    for user_input in test_inputs:
        console.print(f"\n[bold white on blue]  👤 You: {user_input}  [/bold white on blue]\n")
        prediction = agent(user_input=user_input)
        print_turn(prediction)
        console.print("─" * console.width)
    
    
if __name__ == "__main__":
    run_demo()