# prompting defaults

import llama_cpp

##
## system prompts
##

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Always answer if you are able to. If you are unsure about the answer to a question, provide an explanation of why you are unable to answer and suggest additional information that may be useful to include in a subsequent query. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness.'

##
## use llama-cpp-python prompts
##

# alternative names
chat_format_alts = {
    'llama-2': 'llama2',
}

def make_llama_prompt(query, prompt_type='llama-2', system=DEFAULT_SYSTEM_PROMPT):
    # handle alternative names
    prompt_type = chat_format_alts.get(prompt_type, prompt_type.replace('-', '_'))

    # try to get a handler for the prompt type
    maker = getattr(llama_cpp.llama_chat_format, f'format_{prompt_type}', None)
    if maker is None:
        raise ValueError(f'unsupported prompt type: {prompt_type}')

    # execute the handler
    return maker({
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': query},
    })
