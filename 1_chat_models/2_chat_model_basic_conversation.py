# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv
# Import the ChatOpenAI class from langchain_openai to interact with OpenAI's chat models
from langchain_openai import ChatOpenAI
# Import message types for structured conversation
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from a .env file (e.g., your OpenAI API key)
load_dotenv()

# Initialize the ChatOpenAI model with the "gpt-4o" model
model = ChatOpenAI(model="gpt-4o")

# Define a list of messages to simulate a conversation
messages = [
    # SystemMessage: Sets the context or role for the AI. It helps guide the AI's behavior.
    # For example, instructing it to act as a helpful assistant for math problems.
    SystemMessage(content = "You are helpful assistant that can solve math problmes"),
    # HumanMessage: Represents input from the user. This is the actual question or prompt from the human.
    HumanMessage(content = "What is the square root of 81?")
]

# Invoke the model with the messages and get the result
result = model.invoke(messages)
print(f"Result: {result.content}")

# Append the AI's response to the conversation history
messages.append(AIMessage(content = result.content))
# Append a new human question to continue the conversation
messages.append(HumanMessage(content = "What is the square of 144? and what concept involved in earlier question?"))

# Invoke the model again with the updated conversation history
result = model.invoke(messages)
print(f"Result: {result.content}")
