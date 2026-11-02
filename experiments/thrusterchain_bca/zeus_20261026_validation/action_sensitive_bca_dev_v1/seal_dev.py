"""Write a one-time development integrity receipt; this does not authorize execution."""
import json
import diagnostic_runner as d


def main():
    if (d.HERE/'diagnostic_run_01').exists():
        raise RuntimeError('Diagnostic attempt exists; preserve evidence')
    _,_,reference=d.completed()
    record={'schema':'bca.action_sensitive.development_integrity.v1',
            'description':'Byte integrity only; user authorization is external to this receipt',
            'reference_hashes':reference,
            'files':{name:d.sha((d.HERE/name).read_bytes()) for name in d.SEALED_FILES}}
    with (d.HERE/'DEV_SEAL.json').open('x',encoding='utf8',newline='\n') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'status':'development_integrity_sealed','sha256':d.sha((d.HERE/'DEV_SEAL.json').read_bytes())}))


if __name__=='__main__':main()
