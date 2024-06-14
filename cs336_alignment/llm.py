import multiprocessing as mp
from vllm import LLM, SamplingParams
import os

split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]


def run_inference_single_gpu(gpu_id, prompt_list, model_name, sampling_params, tokenizer=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name, tokenizer=tokenizer)
    return llm.generate(prompt_list, sampling_params)

class MultiLLM:
    def __init__(self, model_name: str, num_gpus: int = 8, tokenizer=None):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.tokenizer = tokenizer
    
    def init_and_generate(self, prompts: list[str], sampling_params: SamplingParams):
        split_prompts = split_list(prompts, self.num_gpus)
        inputs = [(i, p, self.model_name, sampling_params, self.tokenizer) for i, p in enumerate(split_prompts)]

        with mp.Pool(processes=self.num_gpus) as pool:
            results = pool.starmap(run_inference_single_gpu, inputs)
        
        outputs = []

        for result in results:
            outputs.extend(result)

        return outputs


