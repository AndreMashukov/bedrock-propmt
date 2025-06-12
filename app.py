import boto3
import json
from botocore.exceptions import ClientError

class WordGenerationApp:
    def __init__(self, region_name="us-east-1"):
        """Initialize the Bedrock client."""
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        # Using Llama 3 70B Instruct model
        self.model_id = "meta.llama3-70b-instruct-v1:0"

    def generate_words(self, language_name):
        """
        Generate words in the specified language using Meta Llama model.
        
        Args:
            language_name (str): The language for word generation
            num_words (int): Number of words to generate (default: 5)
            
        Returns:
            list: List of dictionaries containing word and description
        """
        # Define the prompt for word generation
        prompt = (
            f"Generate 5 unique words that have a random number of characters more than 4 and less than 10 in {language_name} language. "
            "For each word, provide a brief description of its meaning in English with more than a couple of words. "
            "Produce output only in minified JSON array with the keys 'word' and 'description'. Word always must be in lowercase. "
            "Do not include any additional text, explanations, or formatting - only return the JSON array."
        )

        # Format the prompt using Llama 3's instruction format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # Prepare the request payload
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9,
        }

        # Convert to JSON
        request = json.dumps(native_request)

        try:
            # Invoke the model
            response = self.client.invoke_model(modelId=self.model_id, body=request)
            
            # Decode the response
            model_response = json.loads(response["body"].read())
            
            # Extract the generated text
            response_text = model_response["generation"].strip()
            
            # Try to parse the JSON response
            try:
                words_data = json.loads(response_text)
                return words_data
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                # Sometimes the model includes extra text
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    words_data = json.loads(json_str)
                    return words_data
                else:
                    print(f"Could not parse JSON from response: {response_text}")
                    return []
                    
        except ClientError as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return []
        except Exception as e:
            print(f"ERROR: Unexpected error occurred. Reason: {e}")
            return []

    def print_words(self, words_data, language_name):
        """Pretty print the generated words."""
        if not words_data:
            print("No words were generated.")
            return
            
        print(f"\n🌟 Generated {len(words_data)} words in {language_name}:")
        print("=" * 50)
        
        for i, word_info in enumerate(words_data, 1):
            word = word_info.get('word', 'N/A')
            description = word_info.get('description', 'No description available')
            print(f"{i}. {word.upper()}")
            print(f"   📝 {description}")
            print()

def main():
    """Main function to run the word generator."""
    print("🎯 AWS Bedrock Word Generator using Meta Llama")
    print("=" * 50)
    
    # Initialize the app
    app = WordGenerationApp()
    
    while True:
        try:
            # Get user input
            language = input("\nEnter the language for word generation (or 'quit' to exit): ").strip()
            
            if language.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not language:
                print("❌ Please enter a valid language name.")
                continue
            
            print(f"\n🔄 Generating 5 words in {language}...")
            
            # Generate words (always 5)
            result = app.generate_words(language)
            
            # Display results
            app.print_words(result, language)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
