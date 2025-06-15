# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv
# Import the ChatOpenAI class from langchain_openai to interact with OpenAI's chat models
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file (e.g., your OpenAI API key)
load_dotenv()

# Initialize the ChatOpenAI model with the "gpt-4o" model
model = ChatOpenAI(model="gpt-4o")

# Invoke the model with a prompt/question and get the result
result = model.invoke("Add 2+2")

# Print the complete result object returned by the model
print("Complete Result:")
print(result)

# Print only the content (the main text response) from the result
print("Result Content:")
print(result.content)