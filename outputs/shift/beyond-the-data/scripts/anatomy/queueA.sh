while kill -0 1622 2>/dev/null; do sleep 5; done
sh /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy/run_queue.sh 0 critic_beliefs.py 1 2 3 4 --host
sh /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy/run_queue.sh 0 actor_sweep.py 1 2 3 4
sh /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy/run_queue.sh 0 mechanism.py 1 2 3 4
