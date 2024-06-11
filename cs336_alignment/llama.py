from vllm import LLM, SamplingParams
from llm import MultiLLM


def get_llama8b() -> LLM:
    return LLM(model="meta-llama/Meta-Llama-3-8B")

def get_llama8b_multi(num_gpus: int = 8) -> MultiLLM:
    return MultiLLM(model_name="meta-llama/Meta-Llama-3-8B", num_gpus=num_gpus)


# Create a sampling params object, stopping generation on newline.
greedy_sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=1024, stop=["```"]
)


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create an LLM.
    llm = get_llama8b()
    print('Loaded LLM')
    # Generate texts from the prompts. The output is a list of RequestOutput objects # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, greedy_sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()