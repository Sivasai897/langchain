from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import SystemMessage

load_dotenv()

# Initialize the Gemini model for text generation
model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# Define the classification prompt that will determine the sentiment of the feedback
# The model is instructed to return exactly one word: positive, negative, neutral, or escalate
classification_messages =[
    SystemMessage(content = "You are a helpful assistant. Respond with exactly one word: positive, negative, neutral, or escalate."),
    ("human", "classify the sentiment of the following feedback in positive, negative, neutral, or escalate: {text}")
]
classification_prompt_template = ChatPromptTemplate.from_messages(classification_messages)

# Define different response templates for each type of feedback
# Each template has a specific system message and human prompt to generate appropriate responses

# Template for handling positive feedback - generates thank you messages
positive_messages = [
    SystemMessage(content = "You are a helpful assistant that replies for a positive review"),
    ("human","Write a positive thank you message for the following review: {text}")
]
positive_prompt_template = ChatPromptTemplate.from_messages(positive_messages)

# Template for handling negative feedback - generates responses addressing concerns
negative_messages = [
    SystemMessage(content = "You are a helpful assistant that replies for a negative review"),
    ("human", "Generate a response addressing the following negative review: {text}")
]
negative_prompt_template = ChatPromptTemplate.from_messages(negative_messages)

# Template for handling neutral feedback - generates requests for more details
neutral_messages =[
    SystemMessage(content = "You are a helpful assistant that replies for a neutral review"),
    ("human", "Generate a request for more details about this netural review: {text}")
]
neutral_prompt_template = ChatPromptTemplate.from_messages(neutral_messages)

# Template for escalating feedback to human agents when needed
escalation_messages =[
    SystemMessage(content = "You are a helpful assistant that escalates a review to a manager"),
    ("human", "Generate a request for a manager to review the following review: {text}")
]
escalation_prompt_template = ChatPromptTemplate.from_messages(escalation_messages)

# Debug function to print intermediate values during chain execution
def debug_print(x):
    print(f"Input x: {x}")
    return x

# Define the branching logic for different types of feedback
# Each branch checks for specific keywords in the classification result
# The branches are evaluated in order, and the first matching condition is executed
review_branches = RunnableBranch(
    (lambda x: "positive" in x.lower().strip() ,
     positive_prompt_template | model | StrOutputParser() ),
    (lambda x: "negative" in x.lower().strip(),
     negative_prompt_template | model | StrOutputParser() ),
    (lambda x: "neutral" in x.lower().strip(),
     neutral_prompt_template | model | StrOutputParser() ),
    # Default branch for escalation if no other conditions match
    escalation_prompt_template | model | StrOutputParser() 
)

# Create the classification chain that determines the sentiment
classification_chain = classification_prompt_template | model | StrOutputParser()

# Combine the classification and response generation into a single chain
# First, the input is classified, then the appropriate response is generated
chain = classification_chain | review_branches

# Test the chain with a sample positive review
result = chain.invoke({"text":"I love this product! It's amazing!"})

print("--------------------------------")
print(f"Result: {result}")          
print("--------------------------------")














