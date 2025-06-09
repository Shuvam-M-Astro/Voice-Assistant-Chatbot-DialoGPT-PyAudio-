from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import colorama
from colorama import Fore, Style
import time

def setup_chatbot():
    """Initialize the chatbot with model and tokenizer"""
    print(f"{Fore.CYAN}Loading the chatbot model... This may take a moment.{Style.RESET_ALL}")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.YELLOW}Using device: {device}{Style.RESET_ALL}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)
    
    return model, tokenizer, device

def format_system_prompt(role="assistant"):
    """Create a system prompt to give the chatbot a specific personality"""
    system_prompts = {
        "assistant": "You are a helpful and friendly AI assistant. You aim to provide clear, accurate, and helpful responses.",
        "teacher": "You are an educational AI tutor. You explain concepts clearly and encourage learning.",
        "friend": "You are a friendly and casual conversational partner. You engage in light-hearted dialogue.",
    }
    return system_prompts.get(role, system_prompts["assistant"])

def get_response(user_input, model, tokenizer, chat_history_ids=None, device="cpu", max_history_tokens=1000):
    """Generate response with memory management and error handling"""
    try:
        # Encode user input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

        # Manage conversation history
        if chat_history_ids is not None:
            # Truncate history if it's too long
            if chat_history_ids.shape[1] > max_history_tokens:
                # Keep the most recent tokens
                chat_history_ids = chat_history_ids[:, -max_history_tokens:]
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate response with enhanced parameters
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8,
            length_penalty=1.0,
            repetition_penalty=1.2
        )
        
        # Decode and return the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response.strip(), chat_history_ids

    except Exception as e:
        print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
        return "I apologize, but I encountered an error. Let's continue our conversation.", None

def display_welcome_message():
    """Display a formatted welcome message"""
    print(f"\n{Fore.GREEN}{'='*50}")
    print("Welcome to the Enhanced Chatbot!")
    print(f"{'='*50}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Features:{Style.RESET_ALL}")
    print("- Enhanced conversation memory")
    print("- Improved response generation")
    print("- GPU acceleration (if available)")
    print("- Error recovery")
    print(f"\n{Fore.YELLOW}Commands:{Style.RESET_ALL}")
    print("- Type 'quit', 'exit', or 'bye' to end the conversation")
    print("- Type 'clear' to reset the conversation history")
    print("- Type 'help' to see these commands again")
    print(f"\n{Fore.GREEN}{'='*50}{Style.RESET_ALL}\n")

def main():
    """Main chatbot loop with enhanced features"""
    # Initialize colorama for Windows compatibility
    colorama.init()
    
    # Setup
    print(f"{Fore.CYAN}Initializing enhanced chatbot...{Style.RESET_ALL}")
    model, tokenizer, device = setup_chatbot()
    chat_history_ids = None
    
    # Display welcome message
    display_welcome_message()
    
    # Set initial system prompt
    system_prompt = format_system_prompt()
    
    while True:
        # Get user input with colored prompt
        print(f"{Fore.GREEN}You:{Style.RESET_ALL} ", end='')
        user_input = input().strip()
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print(f"\n{Fore.CYAN}Chatbot: Goodbye! Have a great day!{Style.RESET_ALL}")
            break
        elif user_input.lower() == 'clear':
            chat_history_ids = None
            print(f"\n{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}\n")
            continue
        elif user_input.lower() == 'help':
            display_welcome_message()
            continue
        elif not user_input:
            continue
            
        # Generate and display response
        try:
            start_time = time.time()
            response, chat_history_ids = get_response(
                user_input, 
                model, 
                tokenizer, 
                chat_history_ids,
                device
            )
            end_time = time.time()
            
            # Display response with timing information
            print(f"{Fore.CYAN}Chatbot:{Style.RESET_ALL} {response}")
            print(f"{Fore.YELLOW}Response time: {(end_time - start_time):.2f}s{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {str(e)}")
            print("Starting a new conversation.{Style.RESET_ALL}\n")
            chat_history_ids = None

if __name__ == "__main__":
    main()
