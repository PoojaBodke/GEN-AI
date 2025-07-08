"""
Simple chatbot demo.
"""

def rule_based_chatbot(user_input):
    """
    A very basic rule-based chatbot.
    """
    user_input = user_input.lower()
    if "hello" in user_input:
        return "Hi there! How can I help you?"
    elif "bye" in user_input:
        return "Goodbye!"
    else:
        return "I'm not sure how to respond to that."

if __name__ == "__main__":
    print("Rule-based Chatbot (type 'bye' to exit):")
    while True:
        user_input = input("You: ")
        response = rule_based_chatbot(user_input)
        print(f"Bot: {response}")
        if "bye" in user_input.lower():
            break
