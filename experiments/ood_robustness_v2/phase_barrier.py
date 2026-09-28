"""Single-use injected outcome phase. Receipt provenance/runtime remain external.

This is not a supervisor, recovery mechanism or sandbox against arbitrary code.
The actual integration must make this the only route to the accepted panel driver.
"""
import copy
import os
from global_precommit import GATES,append,load,require,sha,verify_seal
from precommit_file import canonical


class OutcomePhase:
    def __init__(self,*,read,write,assert_owner,seal_ref,review_ref,context):
        self._read,self._write,self._owner=read,write,assert_owner
        self._seal_ref,self._review_ref=copy.deepcopy(seal_ref),copy.deepcopy(review_ref)
        self._context=copy.deepcopy(context);self._pid=os.getpid();self._phase='preparing';self._next=0
        try:
            self._check_owner();self._seal=verify_seal(read,self._seal_ref,self._context)
            review=load(read,self._review_ref)
            require(set(review)=={'schema','pair','seal_sha256','execution_declaration_sha256','reviewer_source_sha256','actual_exit','gates'}
                    and review['schema']=='ood-v2-precommit-independent-review-v1'
                    and review['pair']==context['pair'] and review['seal_sha256']==seal_ref['sha256']
                    and review['execution_declaration_sha256']==context['pins']['execution_declaration']
                    and type(review['actual_exit']) is int and review['actual_exit']==0,'Independent review binding failed.')
            sha(review['reviewer_source_sha256'])
            require(type(review['gates']) is dict and set(review['gates'])==GATES
                    and all(v is True for v in review['gates'].values()),'Independent execution gates pending/refused.')
            self._index=load(read,self._seal['index'])
            self._check_owner()
            self._put('phase/claimed',dict(seal=self._seal_ref,review=self._review_ref,pid=self._pid))
            self._phase='ready'
        except BaseException:
            self._phase='failed';raise

    def _check_owner(self):
        require(os.getpid()==self._pid,'Forked phase refused.');self._owner()

    def _put(self,name,value): return append(self._write,self._read,name,value)

    def run_next(self,callback):
        """Fixed host then BCA order; missing states consume nominal panels only."""
        try:
            require(self._phase=='ready' and self._next<512,'Phase not ready, complete or previously failed.')
            self._phase='busy';self._check_owner()
            ordinal=self._next;i=ordinal%256;continuation='host' if ordinal<256 else 'bca'
            # A full scan occurred at phase entry. Recheck current panel plus common
            # roots at each dispatch; accepted exclusive archive ownership is vital.
            for r in (self._seal_ref,self._review_ref,self._seal['index'],self._index['key_binding'],self._index['key_table']): load(self._read,r)
            row=self._index['rows'][i]
            bundle={k:load(self._read,row[k]) if row[k] is not None else None for k in ('capture','bank','warnings')}
            refs=copy.deepcopy(row);name=f'phase/panel{ordinal:03d}'
            self._put(name+'/started',dict(ordinal=ordinal,state_index=i,continuation=continuation,
                                          seal=self._seal_ref,review=self._review_ref,inputs=refs))
            self._check_owner()
            if self._seal['rows'][i]['status']=='missing': result=dict(status='missing_state',state_index=i,continuation=continuation)
            else:
                require(callable(callback),'Panel callback required.')
                result=callback(copy.deepcopy(bundle),continuation)
                require(type(result) is dict,'Panel producer result required.');canonical(result)
            require(self._phase=='busy','Reentrant/failed phase cannot acknowledge completion.')
            self._check_owner()
            for r in (self._seal_ref,self._review_ref,self._seal['index'],self._index['key_binding'],self._index['key_table']): load(self._read,r)
            for k in ('capture','bank','warnings'):
                if row[k] is not None: load(self._read,row[k])
            completed=self._put(name+'/completed',dict(ordinal=ordinal,result=result,scientific_acceptance=False))
            self._next+=1;self._phase='complete' if self._next==512 else 'ready'
            return completed
        except BaseException:
            self._phase='failed';raise
