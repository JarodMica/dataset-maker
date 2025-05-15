# llm_reformatter_script.py
import os
import re
import logging
from math import ceil
import tkinter as tk
from tkinter import filedialog, messagebox

from llama_cpp import Llama
from rapidfuzz import fuzz

# Initialize the module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -------- Configuration --------
FUZZY_THRESHOLD = 70      # Minimum fuzzy matching ratio before warning/defaulting to original
CHUNK_SIZE = 10           # Number of transcript texts per chunk
# --------------------------------

# Note: Removed global logging.basicConfig configuration.
# We'll set up logging in main() so that the log file is written to the same folder as the input file.

def load_model():
    logger.debug("Loading model...")
    model = Llama(
        model_path="DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        n_gpu_layers=-1,  # Use GPU acceleration; set to -1 for all layers if supported.
        seed=1337,        # Set a specific RNG seed for reproducibility.
        n_ctx=2048,       # Increase the context window.
    )
    logger.debug("Model loaded successfully.")
    return model

def remove_thinking_tokens(text):
    logger.debug("Removing thinking tokens from output.")
    # Remove any block between <think> and </think> (including the tags), across multiple lines.
    cleaned = re.sub(r"(?s)<think>.*?</think>", "", text)
    # Remove any stray occurrences of the tokens if they remain.
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()

def build_prompt(transcript_texts):
    """
    Build the prompt using the provided transcript texts.
    Each transcript text (without the file ID) is on its own line.
    The prompt instructs the model explicitly that each transcript is separate
    and must be corrected individually. Then, the prompt is wrapped as:
    
        <｜User｜>{user_input}<｜Assistant｜>
    """
    user_input = """
    You are given sentences that needs to be corrected. Return the sentences back given the guidelines:
1. Quotation Marks:
   - Pay close attention to dialogue. If a portion of the sentence represents spoken words, ensure it is enclosed in quotation marks.
   - If the transcript indicates a speaker’s dialogue, use quotation marks to clearly delineate it.
2. Correct the punctuation.
3. Output Requirements:
   - Return only the corrected sentence(s) with the appropriate punctuation.
   - Do not include any additional text, explanations, or commentary in your output.
   - Each new line represents a transcript that pertains to a specific key.  DO NOT merge sentences that are separated by new lines

Sentences:
    """
    # Append each transcript text on its own line.
    user_input += "\n".join(transcript_texts)
    # Wrap the complete prompt according to the required format.
    wrapped_prompt = f"<｜User｜>{user_input}<｜Assistant｜>"
    logger.debug("Built prompt for chunk with %d transcript lines.", len(transcript_texts))
    return wrapped_prompt

def call_llm(model, prompt):
    logger.debug("Calling LLM with prompt.")
    response_stream = model(
        prompt,
        max_tokens=12000,
        stream=True,
        echo=False,
        temperature=0.6
    )
    
    output = ""
    for chunk in response_stream:
        token_text = chunk["choices"][0]["text"]
        output += token_text
    logger.debug("Received LLM output.")
    return output

def parse_corrected_lines(llm_output):
    logger.debug("Parsing LLM output into lines.")
    cleaned_output = remove_thinking_tokens(llm_output)
    # Split on newlines and filter out any empty lines.
    lines = [line.strip() for line in cleaned_output.splitlines() if line.strip()]
    logger.debug("Parsed %d lines from LLM output.", len(lines))
    return lines

def fuzzy_verify(original_text, corrected_text):
    """
    Uses fuzzy matching to compare the original transcript text with the corrected one.
    Returns the similarity ratio (0 to 100) ignoring punctuation.
    """
    def normalize(text):
        return re.sub(r"[.,!?;:\-\"']", "", text).lower()
    
    norm_orig = normalize(original_text)
    norm_corr = normalize(corrected_text)
    ratio = fuzz.ratio(norm_orig, norm_corr)
    logger.debug("Fuzzy matching ratio for transcript: %d", ratio)
    return ratio

def process_file(input_filepath, output_filepath):
    logger.info("Processing file: %s", input_filepath)
    # Read the input file and split each line into (file_id, transcript_text).
    original_entries = []
    with open(input_filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                logger.error("Line not in expected format (missing '|'): %s", line)
                continue
            file_id = parts[0].strip()
            transcript_text = parts[1].strip()
            original_entries.append((file_id, transcript_text))
    
    total_entries = len(original_entries)
    logger.info("Read %d transcript entries from file.", total_entries)
    corrected_entries = []
    total_chunks = ceil(total_entries / CHUNK_SIZE)
    model = load_model()
    logger.info("Processing %d entries in %d chunk(s).", total_entries, total_chunks)
    
    # Define path for fuzzy match failures log (in the same folder as the output file)
    fuzzy_failures_path = os.path.join(os.path.dirname(output_filepath), "fuzzy_match_failures.txt")
    
    # Open the output and fuzzy failures log files in append mode.
    with open(output_filepath, "a", encoding="utf-8") as outfile, \
         open(fuzzy_failures_path, "a", encoding="utf-8") as fuzzy_fail_log:
        
        # Process entries in chunks.
        for i in range(0, total_entries, CHUNK_SIZE):
            chunk = original_entries[i:i+CHUNK_SIZE]
            # Build prompt using only the transcript_text parts.
            transcript_texts = [entry[1] for entry in chunk]
            prompt = build_prompt(transcript_texts)
            logger.info("Processing chunk %d/%d.", i // CHUNK_SIZE + 1, total_chunks)
            llm_raw_output = call_llm(model, prompt)
            
            # Log the complete LLM raw output for debugging.
            logger.debug("LLM raw output for chunk starting at entry %d:\n%s", i+1, llm_raw_output)
            
            corrected_texts = parse_corrected_lines(llm_raw_output)
            
            if len(corrected_texts) != len(chunk):
                logger.warning("Expected %d output lines but got %d in chunk starting at entry %d.",
                               len(chunk), len(corrected_texts), i+1)
            
            # Process each transcript in the current chunk.
            for j, orig in enumerate(chunk):
                file_id, orig_text = orig

                # Check if the audio segment (i.e. the transcript) is missing.
                if not orig_text.strip():
                    fuzzy_fail_log.write(
                        f"File ID: {file_id}\n"
                        f"Audio segment not found for sentence: {orig_text}\n"
                        f"{'-'*40}\n"
                    )
                    logger.warning("Audio segment not found for entry %s.", file_id)
                    # Default to the original transcript (empty string) and continue.
                    corrected_entries.append((file_id, orig_text))
                    continue

                try:
                    corrected_text = corrected_texts[j]
                except IndexError:
                    logger.error("Missing corrected output for entry %s. Using original text.", file_id)
                    corrected_text = orig_text

                similarity = fuzzy_verify(orig_text, corrected_text)
                # If fuzzy similarity is below the threshold, log the failure and use the original text.
                if similarity < FUZZY_THRESHOLD:
                    logger.warning("Low similarity (%d%%) for entry %s.\nOriginal: %s\nCorrected: %s",
                                   similarity, file_id, orig_text, corrected_text)
                    fuzzy_fail_log.write(
                        f"File ID: {file_id}\n"
                        f"Low similarity: {similarity}%\n"
                        f"Original: {orig_text}\n"
                        f"Corrected (discarded): {corrected_text}\n"
                        f"{'-'*40}\n"
                    )
                    corrected_text = orig_text
                corrected_entries.append((file_id, corrected_text))
            
            # Append the corrected entries from the current chunk to the output file.
            for file_id, corrected_text in corrected_entries[-len(chunk):]:
                outfile.write(f"{file_id} | {corrected_text}\n")
            outfile.flush()  # Ensure the data is written to disk.
            fuzzy_fail_log.flush()
            logger.info("Chunk %d/%d processed and appended to %s.", i // CHUNK_SIZE + 1, total_chunks, output_filepath)
    
    logger.info("Finished processing. Corrected file available at: %s", output_filepath)
    messagebox.showinfo("Done", f"Corrected file written to:\n{output_filepath}")

def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Transcript File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    # Logging will be configured after the file is selected.
    return file_path

def confirm_overwrite(filepath):
    """
    If the output file already exists, ask the user if they want to delete it.
    Returns True if the file was removed (or did not exist) and False if the user chooses not to delete it.
    """
    if os.path.exists(filepath):
        response = messagebox.askyesno(
            "File Exists",
            f"The file '{os.path.basename(filepath)}' already exists in this folder.\nDo you want to delete it and start fresh?"
        )
        if response:
            try:
                os.remove(filepath)
                logger.info("Existing file %s removed.", filepath)
            except Exception as e:
                logger.error("Could not remove the existing file %s: %s", filepath, str(e))
                messagebox.showerror("Error", f"Could not remove the existing file:\n{str(e)}")
                return False
    return True

def main():
    file_path = choose_file()
    if not file_path:
        # Since logging isn't set up yet, we show an error via the messagebox.
        messagebox.showerror("Error", "No file selected. Exiting.")
        return
    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"File '{file_path}' does not exist.")
        return

    # Determine the folder from the selected file and set up logging in that folder.
    folder = os.path.dirname(file_path)
    log_filepath = os.path.join(folder, "logs.txt")
    if not confirm_overwrite(output_filepath):
        logger.info("User chose not to overwrite the existing output file. Appending to existing file.")
        
    logging.basicConfig(
        filename=log_filepath,
        filemode="a",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # Get the root logger and log that logging has been initialized.
    # global logger
    logger = logging.getLogger()
    logger.info("Logging initialized. Log file: %s", log_filepath)

    # Determine the output filepath (train_correct.txt in the same directory as the input file)
    output_filepath = os.path.join(folder, "train_correct.txt")
    
    # Check if the output file already exists and ask the user if they want to delete it.
    if not confirm_overwrite(output_filepath):
        logger.info("User chose not to overwrite the existing output file. Appending to existing file.")
    
    process_file(file_path, output_filepath)

if __name__ == "__main__":
    main()
