# hmm-based_isolated_word_recognizer

This is a HMM Based Isolated Word Recognizer. It contains 3 parts.

1. Sequence scoring using the Forward Algorithm
2. State-level decoding using the Viterbi algorithm
3. Maximum likelihood re-estimation of the transition matrix, using Viterbi training.

## MyHMM class description:

• N_states: An integer value representing the number of total states in the HMM
• pi: A 1-dimensional array of size N_states, where pi[i] = log 𝜋𝑖
• A: A 2-dimensional array of size N_states by N_states representing the transition matrix, where A[i,j] = log 𝑎𝑖𝑗
• labels: A 1-dimensional array of size N_states, where labels[i] indicates the phonetic identity (class index) of state 𝑖. This can be used to retrieve the phonetic label of the state via phone_labels[labels[i]], or to retrieve the observation likelihood 𝑏𝑖 (𝑜𝑡 ) via L[t, labels[i]]

## Function Descriptions:

load_audio_to_mel_spec_tensor: This function takes as input the path to a .wav file, and returns a PyTorch tensor of size (N_frames, 40) representing the log Mel spectrogram of the .wav file, computed over 40 Mel filters.

compute_phone_likelihoods: This function takes as input a deep neural network acoustic model, as well as a PyTorch tensor containing the log Mel spectrogram computed with load_audio_to_melspec_tensor. The output will be a numpy array of shape (N_timesteps, 48), where the first dimension indexes acoustic frames and the second dimension indexes the log likelihoods across the 48 different phoneme classes.

forward: This function should compute 𝑝(𝑂|𝜆), where 𝑋 = 𝑜1, 𝑜2, ...𝑜𝑁 are the acoustic frames representing a speech waveform, and 𝜆 are the acoustic model and HMM parameters. There are three steps.

1. Initialization: 𝛼1(𝑖) = 𝑏𝑖 (𝑜1)𝜋𝑖
2. Induction: 𝛼𝑡+1(𝑖) = [∑𝑁 𝑗=1 𝛼𝑡 (𝑗)𝑎𝑗𝑖 ]𝑏𝑖 (𝑜𝑡+1)
3. Termination: 𝑝(𝑜1, ..., 𝑜𝑇 , 𝑞𝑇 = 𝑠𝑁 |𝜆) = 𝛼𝑇 (𝑁 )

where

• 𝛼𝑡 (𝑖) represents the joint likelihood of having seen the observations up to time 𝑡 𝑜1, 𝑜2, ..., 𝑜𝑡 and being in state 𝑖 at time 𝑡.
• 𝑏𝑖 (𝑜𝑡 ) is the likelihood of emitting observation 𝑜𝑡 in state 𝑖
• 𝑎𝑖𝑗 is the probability of transitioning from state 𝑖 to state 𝑗
• 𝜋𝑖 is the probability of being in state 𝑖 at time 𝑡 = 1
• 𝑞𝑇 = 𝑠𝑁 indicates we are in state 𝑠𝑁 at time 𝑇

The output of the forward algorithm should be a log likelihood, representing log 𝑝(𝑜1, ...𝑜𝑇 , 𝑞𝑇 = 𝑆𝑁 |𝜆) for a given input waveform and the current setting of the HMM parameters.

viterbi: The Viterbi algorithm is very similar to the Forward algorithm, with two notable differences. First, in the induction step, we replace the summation with a max. Second, we keep a backtrace matrix Ψ that we can use to remember the best state at time 𝑡 − 1 that transitioned to some given state at time 𝑡. It contains 4 steps:

1. Initialization: 𝛿1(𝑖) = 𝑏𝑖 (𝑜1)𝜋𝑖 , Ψ1(𝑖) = 0
2. Induction: 𝛿𝑡 (𝑖) = [max𝑗 𝛿𝑡−1(𝑗)𝑎𝑗𝑖 ]𝑏𝑖 (𝑜𝑡 ), Ψ𝑡 (𝑖) = arg max𝑗 𝛿𝑡−1(𝑗)𝑎𝑗𝑖
3. Termination: 𝑞⋆𝑇 = arg max𝑖 𝛿𝑇 (𝑖)
4. Backtrace: 𝑞⋆𝑡 = Ψ𝑡+1(𝑞⋆𝑡+1)

The output of the viterbi algorithm should be the best hidden state sequence for the input observation sequence.
