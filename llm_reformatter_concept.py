# main.py
from llama_cpp import Llama

def load_model():
    """
    Loads the llama.cpp model with the given parameters and returns the model instance.
    """
    model = Llama(
        model_path="DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
        n_gpu_layers=-1,  # Use GPU acceleration; set to -1 for all layers if supported.
        seed=1337,        # Set a specific RNG seed for reproducibility.
        n_ctx=2048,        # Increase the context window.
        
    )
    return model

def stream_inference(model, prompt):
    """
    Generates text from a prompt using streaming tokens and outputs each token as soon as it is generated.
    
    Args:
        model: An instance of the Llama model.
        prompt (str): The prompt string to generate text for.
    """
    # The __call__ method is used here with stream=True so that it returns an iterator.
    # The parameter `echo=True` means that the prompt is included in the output.
    response_stream = model(
        prompt,
        max_tokens=8000,
        stream=True,   # Enable streaming responses.
        echo=True,      # Echo the prompt back in the output.
        temperature=0.6  # Set the temperature to 0.6

    )
    
    # Iterate over the streamed tokens.
    for chunk in response_stream:
        # Each chunk is a dictionary. The generated text is typically in:
        #    chunk["choices"][0]["text"]
        token_text = chunk["choices"][0]["text"]
        # Print without adding a newline each time, and flush the output immediately.
        print(token_text, end="", flush=True)
    print()  # Print a newline after streaming completes.

def main():
    # Load the model
    model = load_model()
    transcript = input("Sentence to correct:")
    user_input = f'''Task:
You are given a sentence that needs to be corrected. Return the sentence back given the guidelines:
1. Quotation Marks:
   - Pay close attention to dialogue. If a portion of the sentence represents spoken words, ensure it is enclosed in quotation marks.
   - If the transcript indicates a speaker’s dialogue, use quotation marks to clearly delineate it.
2. Correct the punctuation.
3. Output Requirements:
   - Return only the corrected sentence(s) with the appropriate punctuation.
   - Do not include any additional text, explanations, or commentary in your output.

Sentence:
{transcript}
'''
    # Define your prompt
    prompt = f'<｜User｜>{user_input}<｜Assistant｜>'

    # Run the streaming inference and print tokens to the terminal
    stream_inference(model, prompt)

if __name__ == "__main__":
    main()
