from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_chatbot():
    # Load model and tokenizer
    print("Loading the chatbot model... This may take a moment.")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return model, tokenizer

def get_response(user_input, model, tokenizer, chat_history_ids=None):
    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history if it exists
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response while limiting history size
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode and return the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def main():
    print("Initializing chatbot...")
    model, tokenizer = setup_chatbot()
    chat_history_ids = None
    
    print("\nChatbot is ready! Type 'quit' to exit.")
    print("\nYou: ", end='')
    
    while True:
        user_input = input()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
            
        if not user_input.strip():
            print("You: ", end='')
            continue
            
        try:
            response, chat_history_ids = get_response(user_input, model, tokenizer, chat_history_ids)
            print(f"Chatbot: {response}")
            print("You: ", end='')
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Let's start a new conversation.")
            chat_history_ids = None
            print("You: ", end='')

if __name__ == "__main__":
    main()
