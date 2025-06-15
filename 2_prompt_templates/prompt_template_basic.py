from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")


# Example 1: Basic Prompt Template
# Use Case: Simple text translation or transformation tasks.
# Why Useful: Allows you to create a reusable template for tasks that require a single input variable.
# How It Works: The template uses a placeholder {text} that gets replaced with the actual input when invoked.
template = "Translate the following text from English to French: {text}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"text": "I love programming in Python"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt)
print("--------------------------------")
print(f"Result: {result.content}")
print("--------------------------------")

# Example 2: Multiple Inputs Prompt Template
# Use Case: Tasks requiring multiple inputs, such as translating multiple messages.
# Why Useful: Enables handling of complex tasks with multiple variables in a single template.
# How It Works: The template uses multiple placeholders ({message1}, {message2}) that get replaced with the actual inputs.
template2 = "Translate the following 2 messages from english to french: Message1: {message1} and Message2: {message2}"
prompt_template2 = ChatPromptTemplate.from_template(template2)

prompt = prompt_template2.invoke({"message1": "Hello", "message2": "How are you?"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt)
print("--------------------------------")
print(f"Result: {result.content}")
print("--------------------------------")


# Example 3: Prompt Template with System and Human Messages
# Use Case: Tasks requiring context or role-setting, such as language-specific translations.
# Why Useful: Allows you to set the AI's role or context (e.g., language preference) and provide specific instructions.
# How It Works: The template uses a list of tuples to define system and human messages, with placeholders for dynamic content.
messages = [
    ("system", "You are a helpful assistant that can answer questions and help with tasks in the following language: {language}"),
    ("human", "Translate the following text from english to {language}: {text}")
]
prompt_template3 = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template3.invoke({"language": "French", "text": "I love programming in Python"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt)
print("--------------------------------")
print(f"Result: {result.content}")
print("--------------------------------")

# This doesn't work, if you want to dynamically append the message then you need to use the above method
# Example 4: Prompt Template with SystemMessage and HumanMessage Objects
# Use Case: Similar to Example 3, but using explicit message objects for clarity and type safety.
# Why Useful: Provides a more structured and type-safe way to define system and human messages.
# How It Works: Uses SystemMessage and HumanMessage objects to define the messages, with placeholders for dynamic content.
messages2 = [
    SystemMessage(content = "You are a helpful assistant that can answer questions and help with tasks in the following language: {language}"),
    HumanMessage(content = "Translate the following text from english to {language}: {text}")
]
prompt_template4 = ChatPromptTemplate.from_messages(messages2)

prompt = prompt_template4.invoke({"language": "French", "text": "I love programming in Python"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt)
print("--------------------------------")
print(f"Result: {result.content}")
print("--------------------------------")

