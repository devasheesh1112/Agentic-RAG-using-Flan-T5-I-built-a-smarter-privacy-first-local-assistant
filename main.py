from processing import initialize_vector_db
from agent import get_answer

# 1. Setup
print("Initializing system...")
retriever = initialize_vector_db("./data")

# 2. Run
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == 'exit': break
    
    response = get_answer(user_query, retriever)
    print(f"\nAI: {response}")