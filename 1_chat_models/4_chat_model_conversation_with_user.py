
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat_history = []

system_message = SystemMessage(content = "You are a helpful assistant that can answer questions and help with tasks")
chat_history.append(system_message)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat...")
        break
    
    chat_history.append(HumanMessage(content = user_input))
    result = model.invoke(chat_history)

    print(f"AI: {result.content}")
    chat_history.append(AIMessage(content = result.content))

    print("\n")


print("Thank you for using the chatbot!")
print(chat_history)
print("-------------Your Chat History-------------")
for message in chat_history:
    print(f"{message.type}: {message.content}")

print("------------------------------------------")
    
