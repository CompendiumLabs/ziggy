import time
import ctypes
import llama_cpp

def llama_decode_tokens(model, tokens, length=None, max_size=32):
    tokens = [tokens] if type(tokens) is int else tokens
    length = len(tokens) if length is None else length
    buffer = (ctypes.c_char * max_size)()
    output = ''
    for i in range(length):
        tid = llama_cpp.llama_token(tokens[i])
        nstr = llama_cpp.llama_token_to_piece(model, tid, buffer, max_size)
        output += buffer[:nstr].decode('utf-8')
    return output

def llama_sample_token(ctx, logits, n_vocab, top_k=40, top_p=0.9, temp=0.4):
    # prepare the token candidates array
    candidates = (llama_cpp.llama_token_data * n_vocab)()
    candidates_p = llama_cpp.llama_token_data_array(candidates, len(candidates), False)
    candidates_r = ctypes.byref(candidates_p)

    # fill in logits data
    for token_id in range(n_vocab):
        candidates[token_id].id = token_id
        candidates[token_id].logit = logits[token_id]
        candidates[token_id].p = 0.0

    # sample new token
    llama_cpp.llama_sample_top_k(ctx, candidates_r, top_k, 1)
    llama_cpp.llama_sample_top_p(ctx, candidates_r, top_p, 1)
    llama_cpp.llama_sample_temp(ctx, candidates_r, temp)
    return llama_cpp.llama_sample_token(ctx, candidates_r)

def llama_load_model(model_path, gpu=False):
    llama_cpp.llama_backend_init(numa=False)
    model_params = llama_cpp.llama_model_default_params()
    model_params.n_gpu_layers = 99 if gpu else 0
    model = llama_cpp.llama_load_model_from_file(model_path.encode('utf-8'), model_params)
    return model

def llama_generate_parallel(model, prompt, n_parallel=1, max_len=256, seed=1234, stream=False, **kwargs):
    # set up context
    ctx_params = llama_cpp.llama_context_default_params()
    ctx_params.seed  = seed
    ctx_params.n_ctx = max_len * n_parallel
    ctx_params.n_batch = max(max_len, n_parallel)
    ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

    # tokenize the prompt
    tokens_list = (llama_cpp.llama_token * max_len)()
    num_tokens = llama_cpp.llama_tokenize(
        model=model,
        text=prompt.encode('utf-8'),
        text_len=len(prompt),
        tokens=tokens_list,
        n_max_tokens=max_len,
        add_bos=False,
        special=False,
    )

    # get memory sizing information
    n_ctx = llama_cpp.llama_n_ctx(ctx)
    n_kv_req = num_tokens + (max_len - num_tokens)*n_parallel

    # create a llama_batch with size 512
    # we use this object to submit token data for decoding
    batch = llama_cpp.llama_batch_init(max(num_tokens, n_parallel), 0, 1)

    # evaluate the initial prompt
    batch.n_tokens = num_tokens
    for i in range(batch.n_tokens):
        batch.token[i] = tokens_list[i]
        batch.pos[i] = i
        batch.seq_id[i][0] = 0
        batch.n_seq_id[i] = 1
        batch.logits[i] = False

    # llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = True

    # execute first decode
    llama_cpp.llama_decode(ctx, batch)

    # assign the system KV cache to all parallel sequences
    # this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for i in range(n_parallel):
        llama_cpp.llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens)

    # remember the batch index of the last token for each parallel sequence
    streams = [''] * n_parallel
    i_batch = [batch.n_tokens - 1] * n_parallel

    # current state of sequences
    n_cur = batch.n_tokens
    n_decode = 0

    while n_cur <= max_len:
        # prepare the next batch
        batch.n_tokens = 0;

        # sample the next token for each parallel sequence / stream
        for i in range(n_parallel):
            # if the stream has already finished
            if i_batch[i] < 0:
                continue

            # get last logits and sample a new token
            n_vocab = llama_cpp.llama_n_vocab(model)
            logits = llama_cpp.llama_get_logits_ith(ctx, i_batch[i])
            new_token_id = llama_sample_token(ctx, logits, n_vocab)

            # is it an end of stream? -> mark the stream as finished
            if new_token_id == llama_cpp.llama_token_eos(ctx) or n_cur == max_len:
                print(f'stream {i} finished at n_cur = {n_cur}')
                i_batch[i] = -1
                continue

            # decode new token and store
            new_token = llama_decode_tokens(model, new_token_id)
            streams[i] += new_token

            # mostly for speed assessment
            if stream:
                print(new_token, end='', flush=True)

            # push this new token for next evaluation
            batch.token[batch.n_tokens] = new_token_id
            batch.pos[batch.n_tokens] = n_cur
            batch.seq_id[batch.n_tokens][0] = i
            batch.n_seq_id[batch.n_tokens] = 1
            batch.logits[batch.n_tokens] = True

            # increment counters
            i_batch[i] = batch.n_tokens
            batch.n_tokens += 1
            n_decode += 1

        # check for done or run next eval
        if batch.n_tokens == 0:
            break
        else:
            llama_cpp.llama_decode(ctx, batch)
            n_cur += 1

    # cleanup code
    llama_cpp.llama_batch_free(batch)
    llama_cpp.llama_free(ctx)
    llama_cpp.llama_backend_free()

    # return outputs
    return streams
