# Identified areas for retraining to improve overall pipeline results

## 1. Goal Distribution Augmentation
The majority of failures identified in T2 instructions are the result of OOB (Out of bounds) or near edge of workspace/boundary issues.
COT prompting preforms the best w/ a grounder success rate of 65% but the entire pipeline struggles with this process which is not surprisng.
CoT is predicting more valid coordinates, w/ no out-of-workspace flags, but the policy is less successful reaching them. This is because COT prompting lands the goal in hard to reach places
instead of just OOB. To address this we propose to retrain the policy w/ a bias sampling (~30%) towards near-edge goals for the policy so the policy becomes more robust/confident with pushing the cube
to targets at the edge. 

Concretely this will look like biasing 30% of HER goal sampling towards goals within 5cm of the edge.

Another idea to imrpove T4 specifically; we noticed that the grounder actually handled T4 instructions by defining "corners" of the workspace to handle dealing w/
obscure instructions like "out of the way." The LLM picks whichever corner it thinks works best and sends the cube there so success is really a coin flip, but to
work with the LLM rather than against it we will retrain with an additional bias towards the corners of the workspace as well so that the policy also becomes more confident
in dealing with workspace corners.

Concretely both of these this will look like a 20/10 bias split; 20% of HER goal sampling towards goals within 5cm of the edge, and 10% in the corners.


## 2. Shaped reward for T3 offset correction
With the original 5cm threshold all T3 instructions were failures, so we raised it to 8cm to make T3 functional which gave us a 100% E2E success on T3. 
The grounder is consistently predicting a goal about 6cm off on the Y-axis based off of run results. Fetchpush uses a default binary reward so instead we will
replace this with our own shaped reward. This will punish rewards for runs that aren't as close to the target vs.
getting closer to the target. This should hopefully push the policy to find the right region even if the goal is slightly off. 

Concretely, for episodes where the cube is more than 4cm from the goal add a distance-proportionality penalty to the reward:

`r_shaped = r_sparse + alpha * (1 - d_block_to_true_goal / d_threshold)`

Keep alpha small, ~0.1 so the policy learns how close it got ideally

## 3. Targeted Replay on failure cases
Take actual failure cases from T2/T3 runs and inject them as seeds for the policy to retrain on at a higher frequency so the policy learns these harder cases.
We inject these hard cases in training early so that these seeds are seen by the policy in the early stages of training hopefully leading to better adaptation.
Use predicted_goal, ground_truth_goal and policy_episodes that we recorded in /results. 

