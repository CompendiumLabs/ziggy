# prompting tools

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

##
## llama
##

# special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_LLAMA_SYSTEM = "You are a helpful assistant. Always answer if you are able to. If you are unsure about the answer to a question, provide an explanation of why you are unable to answer and suggest additional information that may be useful to include in a subsequent query. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness."

# generate a query
def generate_llama(system=None):
    system = DEFAULT_LLAMA_SYSTEM if system is None else system
    return lambda query: f'{B_INST} {B_SYS}\n{system}\n{E_SYS}\n\n{query} {E_INST}'

##
## chatml (openhermes)
##

DEFAULT_CHATML_SYSTEM = 'You are a knowledgable and intelligent assistant. Your purpose is to answer questions posed to you by your users using your general knowledge and the text given below. You should answer the query posed at the end concisely. Do not preface your answer with anything, simply state the answer clearly.'
CHATML_TEMPLATE = "{% for message in messages %}<|im_start|>{{ '' + message['role'] }}\n{% if message['content'] is not none %}{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"

def compile_template(template):
    def raise_exception(message):
        raise TemplateError(message)
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals['raise_exception'] = raise_exception
    return jinja_env.from_string(template)

# generate a query
def generate_chatml(system=None):
    system = DEFAULT_CHATML_SYSTEM if system is None else system
    compiled_template = compile_template(CHATML_TEMPLATE)
    return lambda query: compiled_template.render(messages=[
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': query},
    ])

##
## overall
##

def prompt_generator(prompt_type, system=None):
    if prompt_type == 'llama':
        return generate_llama(system)
    elif prompt_type == 'chatml':
        return generate_chatml(system)
    else:
        return lambda query: query
