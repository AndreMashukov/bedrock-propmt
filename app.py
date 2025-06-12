from aws_cdk import core as cdk
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from aws_cdk import aws_bedrock as bedrock

class WordGenerationApp:
    def __init__(self):
        self.model = bedrock.FoundationModel.from_foundation_model_id(
            self,
            "BedrockModelLlama3",
            bedrock.FoundationModelIdentifier.META_LLAMA_3_70_INSTRUCT_V1,
        )

    def generate_words(self, language_name):
        prompt = (
            "Generate 5 unique words that have a random number of characters more than 4 and less than 10 in {} language. "
            "For each word, provide a brief description of its meaning in English with more than a couple of words. "
            "Produce output only in minified JSON array with the keys word and description. Word always must be in lowercase."
        )

        task = tasks.BedrockInvokeModel(
            self,
            "GenerateWords",
            model=self.model,
            body=sfn.TaskInput.from_object(
                {
                    "prompt": prompt.format(language_name),
                    "max_gen_len": 512,
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            ),
        )

        return task

if __name__ == "__main__":
    app = WordGenerationApp()
    language = input("Enter the language for word generation: ")
    print("Generating words...")
    result = app.generate_words(language)
    print("Generated words:", result)
