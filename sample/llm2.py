from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-Ensemble-v5-GGUF", model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf", model_type="llama", gpu_layers=0)

print(llm("AI is going to"))
