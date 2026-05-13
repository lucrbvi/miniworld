8/03/26:
- got a weird run with a very high loss due to torch.compile(), very weird ... (I rerand w/ the same config w/o torch.compile())
- big batches = cool (gradient accumulation is really good for that)
- torch compile manual: https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0
- last run was really good, need to scale it more, the eval loss did not go up a single time

9/03/26: 
- Will do training on a GH200 on PrimeIntellect, it's cheap and fucking fast for small/medium models
- Gave up on torch compile

10/03/26: 
- flash-attention 3 is cool; Cannot use GH200 on PrimeIntellect :(
- A100 is best for training

11/03/26:
- tried autoresearch, but it turned into slopfest lol

12/03/26:
- I need to redo a run to improve the imprevise model and optimize it, then we can start to labelize and see if it's accurate enough

17/03/26:
- I did not tried to improve it, but now I am back at it :3

18/03/26:
- Got a really good IDM
- Need to implement the labelization pipeline

19/03/26:
- Playing with the parameter-golf challenge from OpenAI

20/03/26:
- I am sick...

22/03/26:
- TODO: Implement the labelization process with the IDM

28/03/26:
- It took me 6 days to implement this fuckass labelization process with the IDM...
- Now we have the clean dataset + the full pipeline

31/03/26:
- First draft for the world model's training script is done

24/04/26:
- I am back, I've check the training scripts and I'm fixing the loss because we are not doing next embedding prediction
- LeWorldModel is a freaking awesome paper

29/04/26:
- Added the rollout steps in the WM loss

30/04/26:
- More grad accumulation = better training (more signal for the optimizer) (the best is apprx. grad_acc=4)
- We have too much data compared to the compute layer (our loop is not really optimized since we are processing images each step for 300 000 frames)
- I am a dumbass, the model generate full white images in inference b.c. it has never seen a full input image + there is no normalization in [0,1] in the training script! + I was giving the same latent N times to the fucking decoder
- Modified the train script to not only mask but also predicting by giving the full frame (we are still masking to force it to be strong)
- Speeding up the training script (cleaning the I/O bullshit we are doing)
- Making the WM attention causal to not brainrot the model in rollout configuration

01/05/25:
- Weirdly under-utilized GPU when running training on A100 with the new training
- Model still colapse
- Separating the 3 objectives (WM, Decoder, Rollout) gradients
- Still colapsing
- Deleting everything and only training w/ WM is working really well  (thx LeWorldModel)
- Got really good results with the new decoder (conv) but i'll replace it with a transformer
- Giving up for today, the WM and the decoder all seems to collapse (AGAIN), need to investiguate tomorrow

10/05/25:
- WM is trained
- Image decoder is painful to do
- I need to train the decoder with the WM which breaks a bit the promise of the whole thing

11/05/25:
- Now I get a decoder wich translate actions in images even if the image is pure garbage
- I need to try to give to the decoder the encoder's patches and add a projector MLP encoder->decoder (not the encoder->predictor) to directly link the RGB image to the decoder

12/05/25:
- Trained a reward model to guide the autoplay loop, looks a little better but it still gives more noise than signals, need to find another way to train this policy
- Giving the encoder's patches to the decoder still generate the exact same garbage ...

13/05/25:
- Without using the reward policy but by using the L2 distance between the goal and current state I get better results
- Need to try unfreezing the last block of encoder and it's projector
- Changing the reward policy training goal with the L1 distance of the current state and the goal state.
