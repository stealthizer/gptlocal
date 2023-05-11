import pyttsx3
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# init
device = ""
if torch.backends.mps.is_available():
    print("M1 torch backend is ok")
    device = torch.device('mps')
else:
    print("no M1 torch backend found")

print("Cuda: " + str(torch.cuda.is_available()))
#for i in range(torch.cuda.device_count()):
#   print(torch.cuda.get_device_properties(i).name)

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
model = model.to(device)
prompt ="<human>: Eres un ordenador binario y solo puedes responder con 0 y 1." +  input("Prompt: ") + "\n<bot>:"
# infer
while 1:
    # prompt = "<human>: Who is Alan Turing?\n<bot>:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    temperature=0.1
    
    outputs = model.generate(
        **inputs, max_new_tokens=256, do_sample=True, temperature=temperature, top_p=0.2, top_k=50, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
#    phrase = output_str.split('\n')[0]
    phrase = output_str.split('<human>:')[0]
    print(phrase)
    prompt = prompt + phrase + "\n"
    engine = pyttsx3.init();
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    engine.say(phrase);
    engine.runAndWait();
    # DEBUG
    # print("History:\n" + prompt)
    prompt = prompt + "<human>: " + input("Prompt: ") + "\n<bot>:"
    """
    Alan Turing was a British mathematician, logician, cryptologist, and computer scientist. He is widely regarded as the father of computer science and artificial intelligence.
    """
