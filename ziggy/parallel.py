import time
import ctypes
import llama_cpp
from llama_cpp._utils import suppress_stdout_stderr

def llama_decode_tokens(model, tokens, length=None, max_size=128):
    tokens = [tokens] if type(tokens) is int else tokens
    length = len(tokens) if length is None else length
    buffer = (ctypes.c_char * max_size)()
    output = ''
    for i in range(length):
        tid = llama_cpp.llama_token(tokens[i])
        nstr = llama_cpp.llama_token_to_piece(model, tid, buffer, max_size)
        output += buffer[:nstr].decode('utf-8')
    return output

def llama_sample_token(ctx, logits, n_vocab, top_k=0, top_p=1.0, temp=1.0):
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
    if top_k > 0:
        llama_cpp.llama_sample_top_k(ctx, candidates_r, top_k, 1)
    if top_p < 1.0:
        llama_cpp.llama_sample_top_p(ctx, candidates_r, top_p, 1)
    llama_cpp.llama_sample_temp(ctx, candidates_r, temp)
    return llama_cpp.llama_sample_token(ctx, candidates_r)

def llama_load_model(model_path, verbose=False, gpu=False):
    with suppress_stdout_stderr(disable=verbose):
        llama_cpp.llama_backend_init(numa=False)
        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = 99 if gpu else 0
        model = llama_cpp.llama_load_model_from_file(model_path.encode('utf-8'), model_params)
    return model

def llama_generate_parallel(model, prompts, max_len=256, seed=1234, stream=False, verbose=False, **kwargs):
    # constants
    n_parallel = len(prompts)
    n_vocab = llama_cpp.llama_n_vocab(model)
    eos_token = llama_cpp.llama_token_eos(model)

    # set up context
    ctx_params = llama_cpp.llama_context_default_params()
    ctx_params.seed = seed
    ctx_params.n_ctx = max_len * n_parallel
    ctx_params.n_batch = 512
    ctx_params.mul_mat_q = 1
    ctx_params.n_threads = 12
    ctx_params.n_threads_batch = 12
    with suppress_stdout_stderr(disable=verbose):
        ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

    # print generation stats
    if verbose:
        print(
            'n_kv_max = %d, is_pp_shared = %d, n_gpu_layers = %d, mmq = %d, n_threads = %d, n_threads_batch = %d\n' % (
                ctx_params.n_ctx, 0, 99, ctx_params.mul_mat_q, ctx_params.n_threads, ctx_params.n_threads_batch
            )
        )

    # tokenize the prompts
    tokens_list, ntoks_list = [], []
    for p in prompts:
        tokens = (llama_cpp.llama_token * max_len)()
        ntoks = llama_cpp.llama_tokenize(
            model=model,
            text=p.encode('utf-8'),
            text_len=len(p),
            tokens=tokens,
            n_max_tokens=max_len,
            add_bos=False,
            special=False,
        )
        tokens_list.append(tokens)
        ntoks_list.append(ntoks)

    # we use this object to submit token data for decoding
    batch = llama_cpp.llama_batch_init(n_parallel, 0, 1)

    # remember the batch index of the last token for each parallel sequence
    streams = [''] * n_parallel
    i_batch = [None] * n_parallel

    # start timing info
    start_time = time.time()
    toks_total = 0

    # run the decoding loop
    for k in range(max_len):
        # prepare the next batch
        batch.n_tokens = 0

        # sample the next token for each parallel sequence / stream
        for i in range(n_parallel):
            # if the stream has already finished
            if i_batch[i] is not None and i_batch[i] < 0:
                continue

            # see if we're still in the prompt
            if k < ntoks_list[i]:
                new_token_id = tokens_list[i][k]
            else:
                # get last logits and sample a new token
                logits = llama_cpp.llama_get_logits_ith(ctx, i_batch[i])
                new_token_id = llama_sample_token(ctx, logits, n_vocab, **kwargs)

                # is it an end of stream? -> mark the stream as finished
                if new_token_id == eos_token:
                    if verbose:
                        print(f'stream {i} finished at k = {k}')
                    i_batch[i] = -1
                    continue

                # decode new token and store
                new_token = llama_decode_tokens(model, new_token_id)
                streams[i] += new_token

            # push this new token for next evaluation
            batch.token[batch.n_tokens] = new_token_id
            batch.pos[batch.n_tokens] = k
            batch.seq_id[batch.n_tokens][0] = i
            batch.n_seq_id[batch.n_tokens] = 1
            batch.logits[batch.n_tokens] = True

            # increment counters
            i_batch[i] = batch.n_tokens
            batch.n_tokens += 1

        # check for done or run next eval
        if batch.n_tokens == 0:
            break
        else:
            llama_cpp.llama_decode(ctx, batch)
            toks_total += batch.n_tokens

    # end timing info
    end_time = time.time()
    if verbose:
        delta = end_time - start_time
        speed = toks_total / delta
        print(f'time: {delta:.3f}s, tokens: {toks_total}, speed: {speed:.3f} tok/s')

    # cleanup code
    llama_cpp.llama_batch_free(batch)
    llama_cpp.llama_free(ctx)

    # return outputs
    return [s.strip() for s in streams]
