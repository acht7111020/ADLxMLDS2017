import numpy as np

def beamsearch(predict_somax, wtoi, k=3, max_len=10):

    eos = wtoi['<eos>']
    empty = 0

    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos
    live_samples = [[empty]]
    live_scores = [0]

    cnt = 0

    while live_k and dead_k < k:
        probs = predict_somax[cnt][0]
        cnt += 1
        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        live_scores = cand_flat[ranks_flat]

        # append the new words to their appropriate live sample
        voc_size = len(probs)
        n_livesample = []
        live_samples = [[unroll(live_samples[r//voc_size])]+[r%voc_size] for r in ranks_flat]

        # live samples that should be dead are...
        zombie = [s[-1] == eos or len(s[0]) >= max_len for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)

        # print(live_k, live_scores, live_samples)
        # print(dead_k, dead_scores, dead_samples)
        # print()

    scores = dead_scores + live_scores
    samples = dead_samples + live_samples
    idx = np.argmin(np.array(scores))
    answer = unroll(samples[idx])

    return answer

def unroll(l):
    x = []

    if type(l) == int:
        return [l]
    for i in l:
        if type(i) == list:
            for v in i :
                x.append(v)
        else: x.append(i)

    return x