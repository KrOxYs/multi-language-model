import time
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

if __name__ == "__main__":
    # Start timer
    start_time = time.time()
    
    # Load model
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

    # Input parameters
    text = "The rapid progress of artificial intelligence (IA) has transformed sectors around the world. From healthcare to finance, AI technologies are used to optimize processes, improve decision-making and offer innovative solutions. One of the most notable advances of AI is the natural language processing (PLN), which allows machines to understand, interpret and generate human language. This has led to the development of applications such as chatbots, language translation tools and automatic content generation. For example, in healthcare, AI-based systems help to perform early diagnoses by analysing medical data and images. In the automotive industry, automotive cars are becoming a real-world processing algorithm."
    source_lang = 'en'
    target_lang = 'es'

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
    print(f"{source_lang.upper()} → {target_lang.upper()}: → {translation[0]}")

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
