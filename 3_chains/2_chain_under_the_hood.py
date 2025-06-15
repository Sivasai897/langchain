from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


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

# Define a function to format the prompt using the prompt template
# This function takes an input dictionary and formats the prompt with the provided values
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# Define a function to invoke the model with the formatted prompt
# This function takes the formatted prompt and invokes the model to get a response
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

# Define a function to parse the model's output into a string
# This function extracts the content from the model's response
parse_output = RunnableLambda(lambda x: x.content)

# Create a chain using RunnableSequence to combine the formatting, model invocation, and output parsing steps
# This chain processes the input through multiple steps to generate a final result
chain = RunnableSequence(first = format_prompt, middle = [invoke_model], last = parse_output)

# Invoke the chain with specific inputs for the topic and text
# This will generate a response based on the provided inputs
result = chain.invoke({"topic":"corporate employees", "text": "Make 3 sarcastic jokes about corporate employees"})

# Print the result of the chain invocation
print("--------------------------------")
print(f"Result: {result}")
print("--------------------------------")

