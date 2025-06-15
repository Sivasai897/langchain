from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Initialize the Google generative AI model with the "gemini-1.5-flash" model
model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# Define a list of messages to create a prompt template
# This template instructs the AI to act as a comedian and make jokes about a given topic
messages = [
    ("system", "You are comedian, make jokes about the following topic: {topic}"),
    ("human", "{text}")
]

# Create a prompt template from the messages
# This template will be used to generate prompts for the AI model
prompt_template = ChatPromptTemplate.from_messages(messages)

# Create a chain that combines the prompt template, the AI model, and the output parser
# The chain processes the input, generates a response, and parses the output into a string
chain = prompt_template | model | StrOutputParser()

# Invoke the chain with specific inputs for the topic and text
# This will generate a response based on the provided inputs
result = chain.invoke({"topic": "AI", "text": " Make 3 sarcastic jokes about AI"})

# Print the result of the chain invocation
print("--------------------------------")   
print(f"Result: {result}")
print("--------------------------------")
