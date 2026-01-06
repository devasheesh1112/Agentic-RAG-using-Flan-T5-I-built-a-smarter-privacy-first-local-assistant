from transformers import pipeline

# Load LLM
llm = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)

def agent_controller(query):
    q = query.lower()
    # Expanded keywords to include conceptual questions
    keywords = [
        "pdf", "document", "data", "summarize", "information", "find", 
        "points", "what is", "types of", "explain", "how does"
    ]
    return "search" if any(word in q for word in keywords) else "direct"

def get_answer(query, retriever):
    action = agent_controller(query)
    source_docs = [] # Initialize an empty list for sources
    
    if action == "search":
        print(f"🕵️ Agent searching documents for: {query}")
        results = retriever.invoke(query)
        source_docs = results # Store the found chunks
        context = "\n".join([r.page_content for r in results])
        prompt = f"Context: {context}\n\nTask: Based on the context, {query}\n\nAnswer:"
    else:
        print(f"🤖 Agent answering directly: {query}")
        prompt = query
    
    response = llm(prompt)[0]["generated_text"]
    
    # Return both the text and the list of source documents
    return response, source_docs

