from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()

# Initialize the Google generative AI model with the "gemini-1.5-flash" model
model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# Define a list of messages to create a prompt template
# This template instructs the AI to act as a helpful assistant and write jokes about a given topic
messages = [
    ("system", "You are a helpful assistant that can write jokes about a given topic: {topic}"),
    ("human", "{text}")
]

# Create a prompt template from the messages
# This template will be used to generate prompts for the AI model
prompt_template = ChatPromptTemplate.from_messages(messages)

# Define a function to convert the output to uppercase
# This function is wrapped in RunnableLambda to make it compatible with the LangChain runnable interface
upper_case_output = RunnableLambda(lambda x: x.upper())

# Define a function to format the final output with a word count
# This function is wrapped in RunnableLambda to make it compatible with the LangChain runnable interface
final_output = RunnableLambda(lambda x: f"Total word count {len(x.split())}\n{x}")

# Create a chain that combines the prompt template, model, output parser, and custom processing steps
# The chain processes the input through multiple steps to generate a final result
chain = prompt_template | model | StrOutputParser() | upper_case_output | final_output

# Invoke the chain with specific inputs for the topic and text
# This will generate a response based on the provided inputs
result = chain.invoke({"topic": "corporate employees", "text": "Make 3 sarcastic jokes about corporate employees"})

# Print the result of the chain invocation
print("--------------------------------")
print(f"Result: {result}")
print("--------------------------------")


