"""Export truthful canonical trajectory playback, not a free-form simulator."""
import hashlib
import json
from pathlib import Path
import numpy as np
from thrusterchain.environment import ThrusterChainEnv
from thrusterchain.canonical import provenance

def export(path):
    x=np.zeros(13); x[10]=.4; x[11]=.2
    cases=[
      ("zero_command",True,np.zeros((100,4))),
      ("counter_current",True,np.full((100,4),-.4)),
      ("over_knee_stall",True,np.full((100,4),-.95)),
      ("same_command_no_stall",False,np.full((100,4),-.95)),
      ("pulsed_command",True,np.repeat(np.where((np.arange(100)//10)%2,-.6,.3)[:,None],4,axis=1)),
    ]
    trajectories=[]
    for name,stall,commands in cases:
        env=ThrusterChainEnv(stall=stall)
        obs,info=env.reset(options={"state":x})
        records=[dict(t=0,state=info["state"].tolist(),observation=obs.tolist())]
        for action in commands:
            previous=env.state
            obs,reward,terminated,truncated,info=env.step(action)
            records.append(dict(t=info["t"],state=info["state"].tolist(),observation=obs.tolist(),
                requested_action=action.tolist(),executed_action=info["command"].tolist(),
                thrust=info["thrust"].tolist(),stalled=info["stalled"].tolist(),reward=reward,
                state_cost=float(np.sum(previous[:10]**2)),
                command_cost=float(.3*np.sum(info["command"]**2)),
                terminated=terminated,truncated=truncated))
        env.close()
        item=dict(name=name,world="stall" if stall else "no_stall",initial_state=x.tolist(),
            action_sequence=commands.tolist(),steps=records,horizon=100)
        item["trajectory_sha256"]=hashlib.sha256(json.dumps(item,sort_keys=True,separators=(",",":"),
                                                           allow_nan=False).encode()).hexdigest()
        trajectories.append(item)
    result=dict(schema="thrusterchain.canonical-playback.v1",
        semantics="finite_horizon; raw canonical physical dynamics; wrapper termination at100",
        provenance=provenance(),units="dimensionless benchmark coordinates; one discrete update per step",
        demo_only=True,experimental_policy_evidence=False,
        consumer_rule="Playback of exported sequences only. Arbitrary slider simulation needs separate parity-verified implementation.",
        trajectories=trajectories)
    with Path(path).open("x",encoding="utf8") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"path":str(Path(path).resolve()),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      "trajectories":len(trajectories),"transitions":500}))
if __name__=="__main__":
    export(Path(__file__).resolve().parent/"contracts"/"trajectory_bank.json")

