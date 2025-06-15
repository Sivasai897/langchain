from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
import datetime


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# The ReAct prompt template is a structured format that helps the agent
# follow a specific pattern of Reasoning and Acting
# It's based on the ReAct framework which combines:
# - Reasoning: The agent thinks about what to do
# - Acting: The agent takes actions using tools
# - Observing: The agent observes the results of its actions
prompt = hub.pull("hwchase17/react")
print("Prompt: ", prompt)

# Define a tool function that returns the current time
# Tools are the building blocks that agents can use to perform actions
# Each tool should have a clear purpose and return value
def get_currrent_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create a list of tools that the agent can use
# Each tool is defined with:
# - name: A unique identifier for the tool
# - func: The actual function to execute
# - description: A clear description of what the tool does
# The description is crucial as the agent uses it to decide when to use the tool
tools = [
    Tool(
        name = "get_current_time",
        func = get_currrent_time,
        description = "Get the current time",
    )
]

# Create a ReAct agent that combines:
# - The language model for reasoning
# - The available tools for actions
# - The ReAct prompt template for structured thinking
# stop_sequence=True ensures the agent stops after completing its task
agent = create_react_agent(
    llm = model,
    tools = tools,
    prompt = prompt,
    stop_sequence= True,
)

# Create an agent executor that:
# - Manages the execution of the agent
# - Handles the interaction between the agent and tools
# - Provides detailed logging of the agent's thought process when verbose=True
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True)

# Invoke the agent with a question
# The agent will:
# 1. Understand the question
# 2. Decide which tool to use
# 3. Execute the tool
# 4. Return the result
result = agent_executor.invoke({"input":"What is the current time now?"})

print("Result: ", result)









