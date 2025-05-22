
class phi:
    def __init__(self, version="3.5"):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        if version == "4":
            # transformers==4.48.2
            # backoff==2.2.1
            # peft==0.13.2
            model_id = "microsoft/Phi-4-multimodal-instruct"
        elif version == "3.5":
            # transformers==4.43.0
            # accelerate==0.30.0
            model_id = "microsoft/Phi-3.5-vision-instruct" 
        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'    
        )
        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            # num_crops=4
        )
        self.generation_config = GenerationConfig.from_pretrained(model_id)

        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

    def query_1(self, query_list, verbose=False): 
        images = []
        placeholder = ""
        for i in range(2): 
            images.append(query_list[i])
            placeholder += f"<|image_{i+1}|>\n"

        messages = [
            {"role": "user", "content": placeholder+query_list[2]},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = { 
            "max_new_tokens": 1000, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args,
            generation_config=self.generation_config
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0] 

        if verbose:
            print(response)
        return response

    def query_2(self, query_list, summary_prompt, verbose=False): 
        images = []
        placeholder = ""
        for i in range(2): 
            images.append(query_list[i])
            placeholder += f"<|image_{i+1}|>\n"

        messages = [
            {"role": "user", "content": placeholder+query_list[2]},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = { 
            "max_new_tokens": 1000, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args, 
            generation_config=self.generation_config
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0] 

        if verbose:
            print(response)
        
        # summary
        prompt = f'{self.user_prompt}{summary_prompt.format(response)}{self.prompt_suffix}{self.assistant_prompt}'
        inputs = self.processor(prompt, images=None, return_tensors="pt").to("cuda:0")

        generate_ids = self.model.generate(
           **inputs,
            max_new_tokens=10,
            generation_config=self.generation_config,
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        
        if verbose:
           print("summary: ", response)
        return response