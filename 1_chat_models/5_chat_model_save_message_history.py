from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()



"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

project_id = "angchainlearning-20f80"
collection_name = "chat_history"
session_id = "first_session"

client = firestore.Client(project=project_id)

chat_history = FirestoreChatMessageHistory(
    session_id=session_id,
    collection=collection_name,
    client=client
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

print("--------------------------------")
print("Chat History:")
print(chat_history.messages)
print("--------------------------------")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat...")
        break

    chat_history.add_user_message(user_input)

    result = model.invoke(chat_history.messages)

    print(f"AI: {result.content}")

    chat_history.add_ai_message(result.content)

    print("\n")

print("Thank you for using the chatbot!")   

print("-------------Your Chat History-------------")

for message in chat_history.messages:
    print(f"{message.type}: {message.content}")

print("------------------------------------------")



