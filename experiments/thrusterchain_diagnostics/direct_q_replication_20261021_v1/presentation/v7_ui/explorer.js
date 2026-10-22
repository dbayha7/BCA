/* Saved-evidence explorer. No model inference, new sampling or network access. */
(() => {
  'use strict';
  const root = document.getElementById('results-explorer');
  const dataNode = document.getElementById('xe-data');
  if (!root || !dataNode) return;
  const data = JSON.parse(dataNode.textContent);
  const methods = ['raw_q','live_width','matched_distance','shuffled_width','constant_offset'];
  const names = {raw_q:'Raw Q1',live_width:'Q1 − width',matched_distance:'Q1 − matched distance',shuffled_width:'Q1 − shuffled width',constant_offset:'Q1 − state-constant offset'};
  const contexts = {stall_low:'Stalling · low load',stall_high:'Stalling · high load',monotone_low:'Monotone · low load',monotone_high:'Monotone · high load'};
  const notes = {
    raw_q:'The first critic chooses the candidate with the largest Q1. This is the baseline for every displayed change.',
    live_width:'Subtract the native width U with its frozen coefficient of 1. Larger differences in width can change the winning candidate.',
    matched_distance:'Subtract kD, with k = 0.519506430786272 fixed before this bank. D is the mean squared command distance from the behavior mode.',
    shuffled_width:'Use the one saved within-state shuffle of widths and usability flags. The values stay fixed; their placement across candidates changes.',
    constant_offset:'Subtract the same mean width from all candidates at a state. This changes the score level while preserving the order and chosen command.'
  };
  const $ = id => document.getElementById(id);
  const mean = values => values.length ? values.reduce((a,b) => a+b,0)/values.length : null;
  const fmt = (x,n=4) => Number.isFinite(x) ? x.toFixed(n).replace(/^-0\.(0+)$/, '0.$1') : 'Unavailable';
  const sign = (x,n=4) => Number.isFinite(x) ? (x>0 ? '+' : '') + fmt(x,n) : 'Unavailable';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const outcome = value => value > 0 ? 'gain' : value < 0 ? 'loss' : 'same';
  const filterStates = (context,time) => data.states.filter(s => s.valid && (context==='all'||s.context===context) && (time==='all'||s.time===Number(time)));
  function summarize(states,method) {
    const valid=states.filter(s => s.valid && s.methods[method]);
    const deltas=valid.map(s => s.methods[method].delta_vs_raw);
    const ranking=valid.filter(s => Number.isFinite(s.methods[method].concordance) && Number.isFinite(s.methods.raw_q.concordance));
    return {n:valid.length,gain:mean(deltas),improved:deltas.filter(v=>v>0).length,harmed:deltas.filter(v=>v<0).length,unchanged:deltas.filter(v=>v===0).length,
      concordanceChange:mean(ranking.map(s=>s.methods[method].concordance-s.methods.raw_q.concordance)),rankingN:ranking.length,
      changedCommands:valid.filter(s=>s.methods[method].changed_command).length};
  }
  function sortedCandidates(state,method,sort) {
    const list=state.candidates.slice();
    const difference=(a,b)=>sort==='score'?b.scores[method]-a.scores[method]:sort==='truth'?b.true_policy_q-a.true_policy_q:sort==='width'?a.width-b.width:a.id-b.id;
    return list.sort((a,b)=>difference(a,b)||a.id-b.id);
  }
  function csvFor(state,method) {
    const fields=['state_id','candidate_id','context','time','method','raw_q1','penalty','score','measured_qpi','signed_q1_error','width','distance','support','action_0','action_1','action_2','action_3'];
    const quote=v=>'"'+String(v).replace(/"/g,'""')+'"';
    return [fields,...state.candidates.map(c=>[state.id,c.id,state.context,state.time,method,c.q1,c.q1-c.scores[method],c.scores[method],c.true_policy_q,c.signed_q1_error,c.width,c.distance,c.support_class,...c.action])].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  }
  const initial=()=>({context:'all',time:'all',method:'live_width',stateId:24,candidateId:7,sort:'id',view:'choices',ingredients:false});
  let state=initial();
  const current=()=>data.states.find(s=>s.id===state.stateId);
  function setText(id,value){$(id).textContent=value;}
  function announce(text){setText('xe-status',text);}
  function selectState(id){state.stateId=id;state.candidateId=current().methods[state.method].chosen_index;render();}
  function revealDetail(){const heading=$('xe-detail-heading');heading.focus({preventScroll:true});heading.scrollIntoView({block:'start',behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});}
  function pickCandidate(id,source){
    state.candidateId=id;renderDetail();
    const target=$(source==='plot'?'xe-point-'+id:'xe-candidate-'+id);
    if(target) target.focus({preventScroll:true});
    announce('Candidate '+id+' selected. '+names[state.method]+' score '+fmt(current().candidates.find(c=>c.id===id).scores[state.method],6)+'.');
  }
  function render(){
    const filtered=filterStates(state.context,state.time);
    if(!filtered.some(s=>s.id===state.stateId)){
      state.stateId=filtered[0].id;state.candidateId=current().methods[state.method].chosen_index;
    }
    $('xe-context').value=state.context;$('xe-time').value=state.time;$('xe-sort').value=state.sort;
    root.querySelectorAll('input[name="xe-method"]').forEach(input=>{input.checked=input.value===state.method;});
    methods.forEach(method=>setText('xe-mean-'+method,sign(summarize(filtered,method).gain,4)));
    const summary=summarize(filtered,state.method);
    setText('xe-method-note',notes[state.method]);
    setText('xe-gain',sign(summary.gain,6));
    setText('xe-outcomes',summary.improved+' / '+summary.harmed+' / '+summary.unchanged);
    setText('xe-count',summary.n+' states · '+summary.changedCommands+' changed commands');
    setText('xe-ordering',sign(summary.concordanceChange*100,4)+' pp');
    setText('xe-scope',(state.context==='all'?'All contexts':contexts[state.context])+' · '+(state.time==='all'?'all assigned times':'assigned time '+state.time)+'. Equal-state means over '+summary.n+' states; '+summary.rankingN+' states contribute to pair ordering. Point estimates only.');
    const allowed=new Set(filtered.map(s=>s.id));
    data.states.forEach(s=>{
      const tile=$('xe-state-'+s.id),delta=s.methods[state.method].delta_vs_raw;
      tile.hidden=!allowed.has(s.id);tile.setAttribute('aria-pressed',String(s.id===state.stateId));tile.setAttribute('data-outcome',outcome(delta));
      tile.setAttribute('aria-label','State '+s.id+', '+contexts[s.context]+', time '+s.time+', '+(delta===0?'same return':outcome(delta)+' '+fmt(Math.abs(delta),6)));
      setText('xe-tile-value-'+s.id,delta===0?'Same':sign(delta,2));
    });
    ['gain','loss','same'].forEach(kind=>{$('xe-case-'+kind).disabled=!filtered.some(s=>outcome(s.methods[state.method].delta_vs_raw)===kind);});
    const i=filtered.findIndex(s=>s.id===state.stateId);$('xe-prev').disabled=i===0;$('xe-next').disabled=i===filtered.length-1;
    renderDetail();
  }
  function renderDetail(){
    const s=current(),selected=s.methods[state.method],raw=s.methods.raw_q;
    root.setAttribute('data-ingredients',String(state.ingredients));$('xe-ingredients').checked=state.ingredients;
    const delta=selected.delta_vs_raw;
    setText('xe-state-meta',contexts[s.context]+' / time '+s.time+' / '+(100-s.time)+' steps remaining');
    setText('xe-detail-heading','State '+s.id);
    const choice=selected.changed_command?'selects candidate '+selected.chosen_index+' instead of raw Q1’s candidate '+raw.chosen_index:'keeps the raw-Q1 command (candidate '+selected.chosen_index+')';
    const consequence=delta===0?'Its continuation return is unchanged.':'Its continuation return is '+fmt(Math.abs(delta),6)+' '+(delta>0?'higher.':'lower.');
    setText('xe-decision',names[state.method]+' '+choice+'. '+consequence);
    setText('xe-raw-choice','Candidate '+raw.chosen_index);setText('xe-raw-return','Qπ = '+fmt(raw.chosen_true_q,6));
    setText('xe-rule-label',names[state.method]);setText('xe-rule-choice','Candidate '+selected.chosen_index);setText('xe-rule-return','Qπ = '+fmt(selected.chosen_true_q,6));
    setText('xe-actor-return','Qπ = '+fmt(s.actor_true_q,6));
    $('xe-view-choices').setAttribute('aria-pressed',String(state.view==='choices'));
    $('xe-view-errors').setAttribute('aria-pressed',String(state.view==='errors'));
    $('xe-error-panel').hidden=state.view!=='errors';
    setText('xe-correlation','This state: Pearson(width, signed Q1 error) = '+fmt(s.rho_width,6)+'. Pair ordering: '+fmt(raw.concordance*100,2)+'% raw → '+fmt(selected.concordance*100,2)+'% selected rule. Signed error always uses raw Q1.');
    const supports={common_on_grid:'Common grid',rare_on_grid:'Rare grid',unsupported:'Off grid'};
    $('xe-candidate-body').innerHTML=sortedCandidates(s,state.method,state.sort).map(c=>{
      const badges=[];if(c.id===selected.chosen_index)badges.push('Chosen');if(c.id===raw.chosen_index)badges.push('Raw');if(c.id===6)badges.push('Actor');
      return '<tr data-selected="'+(c.id===state.candidateId)+'"><th scope="row"><button type="button" id="xe-candidate-'+c.id+'" data-candidate="'+c.id+'" aria-pressed="'+(c.id===state.candidateId)+'" aria-label="Inspect candidate '+c.id+'">'+c.id+'</button>'+badges.map(v=>'<span class="xe-badge">'+v+'</span>').join('')+'</th><td class="xe-ingredient">'+fmt(c.q1)+'</td><td class="xe-ingredient">'+fmt(c.q1-c.scores[state.method])+'</td><td>'+fmt(c.scores[state.method])+'</td><td>'+fmt(c.true_policy_q)+'</td><td>'+supports[c.support_class]+'</td></tr>';
    }).join('');
    const c=s.candidates.find(c=>c.id===state.candidateId);
    const precision='Shown to six decimals; selection uses full saved precision.';
    $('xe-candidate-detail').innerHTML='<p><b>Candidate '+c.id+': score arithmetic</b></p><p>'+fmt(c.q1,6)+' − '+fmt(c.q1-c.scores[state.method],6)+' = <b>'+fmt(c.scores[state.method],6)+'</b>. '+precision+'</p><p>Measured Qπ: <b>'+fmt(c.true_policy_q,6)+'</b> · signed Q1 error: <b>'+sign(c.signed_q1_error,6)+'</b> · native width: '+fmt(c.width,6)+'.</p><p>Command: ['+c.action.map(x=>fmt(x,6)).join(', ')+']. '+escape(supports[c.support_class])+', behavior mass '+c.beta.toPrecision(5)+'. Width '+(c.width_usable?'usable':'refused')+'; finite width does not establish action support.</p>';
    renderPlot(s);
  }
  function renderPlot(s){
    const xs=s.candidates.map(c=>c.width),ys=s.candidates.map(c=>c.signed_q1_error);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const xp=Math.max((xmax-xmin)*.12,.01),yp=Math.max((ymax-ymin)*.12,.1);
    const xlo=xmin-xp,xhi=xmax+xp,ylo=ymin-yp,yhi=ymax+yp;
    const X=x=>70+(x-xlo)/(xhi-xlo)*570,Y=y=>260-(y-ylo)/(yhi-ylo)*225;
    let svg='';
    for(let i=0;i<=4;i++){
      const x=xlo+(xhi-xlo)*i/4,y=ylo+(yhi-ylo)*i/4;
      svg+='<line x1="70" y1="'+Y(y)+'" x2="640" y2="'+Y(y)+'" stroke="var(--rule)"/><text x="60" y="'+(Y(y)+4)+'" text-anchor="end" fill="currentColor" font-size="12">'+fmt(y,1)+'</text><text x="'+X(x)+'" y="281" text-anchor="middle" fill="currentColor" font-size="12">'+fmt(x,2)+'</text>';
    }
    if(ylo<0&&yhi>0)svg+='<line x1="70" y1="'+Y(0)+'" x2="640" y2="'+Y(0)+'" stroke="var(--muted)" stroke-dasharray="4 4"/>';
    svg+='<text x="70" y="18" fill="currentColor" font-size="13">Signed error, Q1 − Qπ ↑</text><text x="355" y="310" fill="currentColor" text-anchor="middle" font-size="13">Native uncertainty width, U →</text>';
    s.candidates.forEach(c=>{
      const chosen=c.id===s.methods[state.method].chosen_index,raw=c.id===s.methods.raw_q.chosen_index;
      const color=chosen?'var(--accent)':raw?'var(--ink)':'var(--muted)';
      const label='Candidate '+c.id+(chosen?', chosen':'')+(raw?', raw-Q1 choice':'')+(c.id===6?', original actor':'')+', width '+fmt(c.width,6)+', signed error '+fmt(c.signed_q1_error,6);
      if(c.id===6)svg+='<circle cx="'+X(c.width)+'" cy="'+Y(c.signed_q1_error)+'" r="12" fill="none" style="stroke:var(--ink);stroke-width:1.5;pointer-events:none"/>';
      svg+='<circle id="xe-point-'+c.id+'" data-candidate="'+c.id+'" tabindex="0" role="button" aria-pressed="'+(c.id===state.candidateId)+'" aria-label="'+escape(label)+'" cx="'+X(c.width)+'" cy="'+Y(c.signed_q1_error)+'" r="'+(c.id===state.candidateId?8:6)+'" fill="'+color+'"><title>'+escape(label)+'</title></circle><text x="'+(X(c.width)+9)+'" y="'+(Y(c.signed_q1_error)-8)+'" font-size="12" fill="currentColor" pointer-events="none">'+c.id+(chosen?' Chosen':raw?' Raw':'')+'</text>';
    });
    $('xe-scatter-content').innerHTML=svg;
  }
  $('xe-state-grid').innerHTML=data.states.map(s=>'<button type="button" class="xe-state" id="xe-state-'+s.id+'" data-state="'+s.id+'" aria-pressed="false"><small>State '+s.id+' · t'+s.time+'</small><strong id="xe-tile-value-'+s.id+'">—</strong></button>').join('');
  root.querySelectorAll('input[name="xe-method"]').forEach(input=>input.addEventListener('change',()=>{
    if(!input.checked)return;state.method=input.value;state.candidateId=current().methods[state.method].chosen_index;render();announce(names[state.method]+' selected. '+$('xe-count').textContent+'.');
  }));
  ['context','time'].forEach(key=>$('xe-'+key).addEventListener('change',()=>{state[key]=$('xe-'+key).value;render();announce($('xe-scope').textContent);}));
  $('xe-reset').addEventListener('click',()=>{state=initial();render();announce('Reset to all 32 states, subtract width, state 24.');});
  $('xe-state-grid').addEventListener('click',event=>{const button=event.target.closest('[data-state]');if(button){selectState(Number(button.getAttribute('data-state')));revealDetail();announce($('xe-decision').textContent);}});
  ['gain','loss','same'].forEach(kind=>$('xe-case-'+kind).addEventListener('click',()=>{
    const options=filterStates(state.context,state.time).filter(s=>outcome(s.methods[state.method].delta_vs_raw)===kind);
    options.sort((a,b)=>kind==='gain'?b.methods[state.method].delta_vs_raw-a.methods[state.method].delta_vs_raw:kind==='loss'?a.methods[state.method].delta_vs_raw-b.methods[state.method].delta_vs_raw:a.id-b.id);
    if(options.length){selectState(options[0].id);revealDetail();announce($('xe-decision').textContent);}
  }));
  ['prev','next'].forEach(direction=>$('xe-'+direction).addEventListener('click',()=>{
    const list=filterStates(state.context,state.time),i=list.findIndex(s=>s.id===state.stateId)+(direction==='next'?1:-1);
    if(list[i]){selectState(list[i].id);announce($('xe-decision').textContent);}
  }));
  ['choices','errors'].forEach(view=>$('xe-view-'+view).addEventListener('click',()=>{state.view=view;renderDetail();announce(view==='errors'?'Showing width versus signed error for state '+state.stateId+'.':'Showing candidate comparison.');}));
  $('xe-ingredients').addEventListener('change',()=>{state.ingredients=$('xe-ingredients').checked;renderDetail();announce(state.ingredients?'Raw Q1 and penalty columns shown.':'Showing the compact candidate table.');});
  $('xe-sort').addEventListener('change',()=>{state.sort=$('xe-sort').value;renderDetail();announce('Candidate table reordered. Saved winning candidate is unchanged.');});
  $('xe-candidate-body').addEventListener('click',event=>{const button=event.target.closest('[data-candidate]');if(button)pickCandidate(Number(button.getAttribute('data-candidate')),'table');});
  const plotChoice=event=>{const point=event.target.closest('[data-candidate]');if(point)pickCandidate(Number(point.getAttribute('data-candidate')),'plot');};
  $('xe-scatter').addEventListener('click',plotChoice);
  $('xe-scatter').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();plotChoice(event);}});
  $('xe-export').addEventListener('click',()=>{
    const blob=new Blob([csvFor(current(),state.method)],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');
    link.href=url;link.download='ThrusterChain_state_'+state.stateId+'_'+state.method+'.csv';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);announce('Downloaded the 12 saved candidates for state '+state.stateId+'.');
  });
  // A small namespaced audit surface supports pure arithmetic and controller tests.
  window.ThrusterExplorer=Object.freeze({filterStates,summarize,sortedCandidates,csvFor,getState:()=>({...state}),data});
  $('xe-app').hidden=false;
  render();
})();
