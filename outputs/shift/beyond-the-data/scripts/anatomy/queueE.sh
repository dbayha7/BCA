while pgrep -f "critic_beliefs.py 2" >/dev/null; do sleep 5; done
sh /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy/run_queue.sh 0 critic_beliefs.py 4 --host
