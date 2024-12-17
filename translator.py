import time
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

if __name__ == "__main__":
    # Start timer
    start_time = time.time()
    
    # Load model
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

    # Input parameters
    text = "Hello nama saya raifandi saya tinggal di indonesia lebih tepatnya di jakarta"
    source_lang = 'en'
    target_lang = 'id'

    # Set source language
    tokenizer.src_lang = source_lang
    
    # Encode the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    
    # Decode and print translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"{source_lang.upper()} → {target_lang.upper()}: {text} → {translation[0]}")

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
