from langchain_community.llms.ollama import Ollama

# Initialize the Llama3 model via Ollama
try:
    llm = Ollama(model="llama3")
    print("Llama3 model initialized successfully!")
except Exception as e:
    print(f"Failed to initialize Llama3 model: {e}")
    exit(1)

# Test simple response generation
try:
    user_input = ["Bonjour, comment ça va ?"]  # Adjust input to be a list
    response = llm.generate(user_input)
    print("User input:", user_input[0])
    print("Response from Llama3:", response)
except Exception as e:
    print(f"Failed to generate response: {e}")
