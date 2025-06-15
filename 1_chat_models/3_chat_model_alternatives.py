# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv
# Import the ChatOpenAI class from langchain_openai to interact with OpenAI's chat models
from langchain_openai import ChatOpenAI
# Import the ChatGoogleGenerativeAI class from langchain_google_genai to interact with Google's generative AI models
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file (e.g., your OpenAI and Google API keys)
load_dotenv()

# Initialize the OpenAI chat model with the "gpt-4o" model
model_openai = ChatOpenAI(model="gpt-4o")
# Initialize the Google generative AI model with the "gemini-1.5-flash" model
model_google = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Invoke the OpenAI model with a prompt and get the result
result_openai = model_openai.invoke("What is the capital of India?")
# Invoke the Google model with the same prompt and get the result
result_google = model_google.invoke("What is the capital of India?")

# Print the result from the OpenAI model
print(f"OpenAI Result: {result_openai.content}")
# Print the result from the Google model
print(f"Google Result: {result_google.content}")

