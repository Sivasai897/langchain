from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.messages import SystemMessage

load_dotenv()

# Initialize the Google generative AI model with the "gemini-1.5-flash" model
model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# Define a list of messages to create a prompt template
# This template instructs the AI to act as a helpful assistant and write features for a given product
messages = [
    SystemMessage(content = "You are a helpful assistant that helps in writing features for a product"),
    ("human", "Write the features for the following product: {product}")
]

# Create a prompt template from the messages
# This template will be used to generate prompts for the AI model
prompt_template = ChatPromptTemplate.from_messages(messages)

# Define a function to analyze the pros of a product's features
# This function creates a prompt template to identify the pros of the given features
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        SystemMessage(content = "You are a helpful assistant that can analyze the features of a product and identify the pros"),
        ("human", "Features: {features}\n\nIdentify the pros:")
    ])

    return pros_template.format_prompt(features = features)

# Define a function to analyze the cons of a product's features
# This function creates a prompt template to identify the cons of the given features
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        SystemMessage(content = "You are a helpful assistant that can analyze the features of a product and identify the cons"),
        ("human", "Features: {features}\n\nIdentify the cons:")
    ])

    return cons_template.format_prompt(features = features)

# Wrap the analyze_pros function in RunnableLambda to make it compatible with the LangChain runnable interface
analyze_pros_lambda = RunnableLambda(lambda x: analyze_pros(x))

# Wrap the analyze_cons function in RunnableLambda to make it compatible with the LangChain runnable interface
analyze_cons_lambda = RunnableLambda(lambda x: analyze_cons(x))

# Create a chain for analyzing the pros of the product features
# This chain processes the input through the analyze_pros function, invokes the model, and parses the output
pros_branch_chain = analyze_pros_lambda | model | StrOutputParser()

# Create a chain for analyzing the cons of the product features
# This chain processes the input through the analyze_cons function, invokes the model, and parses the output
cons_branch_chain = analyze_cons_lambda | model | StrOutputParser()

# Create a parallel chain using RunnableParallel to run the pros and cons analysis in parallel
# This allows both analyses to be performed simultaneously, improving efficiency
parallel_chain = RunnableParallel(branches = {"pros": pros_branch_chain, "cons": cons_branch_chain})

# Define a function to format the final output with the pros and cons
# This function is wrapped in RunnableLambda to make it compatible with the LangChain runnable interface
final_output = RunnableLambda(lambda x: f"Pros: \n{x['branches']['pros']}\nCons: {x['branches']['cons']}")

# Create a chain that combines the prompt template, model, parallel chain, and final output formatting
# The chain processes the input through multiple steps to generate a final result
chain = prompt_template | model | parallel_chain | final_output

# Invoke the chain with a specific product input
# This will generate a response based on the provided product
result = chain.invoke({"product": "Iphone 15"})

# Print the result of the chain invocation
print("--------------------------------")
print(f"Result: {result}")
print("--------------------------------")

