# prompting defaults

##
## llama
##

# special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_LLAMA_SYSTEM = "You are a helpful assistant. Always answer if you are able to. If you are unsure about the answer to a question, provide an explanation of why you are unable to answer and suggest additional information that may be useful to include in a subsequent query. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness."

##
## chatml (openhermes)
##

DEFAULT_CHATML_SYSTEM = 'You are a knowledgable and intelligent assistant. Your purpose is to answer questions posed to you by your users using your general knowledge and the text given below. You should answer the query posed at the end concisely. Do not preface your answer with anything, simply state the answer clearly.'
CHATML_TEMPLATE = "{% for message in messages %}<|im_start|>{{ '' + message['role'] }}\n{% if message['content'] is not none %}{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
