#!/bin/bash
# usage: run.sh <label> <extra demo args...>; bounded non-live run with GPU utilisation sampling
label=$1; shift
O=${OUT:-/tmp/gpu-saturation}/$label; rm -rf $O $O.util $O.log
mkdir -p "$(dirname "$O")"; cd "$(dirname "$0")/../../.."
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits -lms 500 > $O.util &
smi=$!
timeout 280 pixi run python tomography/demo.py --gpu-stress --algorithm fbp --network auto --transport-batch 16 \
  --pixels 256 --slices 256 --no-saving --no-decompression --no-verify-volumes --update-projections 48 \
  --scans 12 --scan-period 0 --output $O "$@" > $O.log 2>&1
echo "exit=$?"
kill $smi
python3 - $O $O.util <<'P'
import json,sys,statistics as st
u=[tuple(map(float,l.split(','))) for l in open(sys.argv[2]) if l.strip()]
n=len(u); mid=u[n//4:max(n//4+1,n-n//6)]
print('gpu util% median/mean (middle of run):', st.median(x[0] for x in mid), round(st.mean(x[0] for x in mid),1), 'power W', round(st.mean(x[1] for x in mid),1), 'peak MiB', max(x[2] for x in u), 'samples', n)
import glob
for f in glob.glob(sys.argv[1]+'/summary.json')+glob.glob(sys.argv[1]+'/status.json'):
    s=json.load(open(f)); print(f.split('/')[-1], {k:v for k,v in s.items() if k in('elapsed_seconds','acquisition_seconds','volumes','volumes_per_second','throughput','completed_scans','phase')})
    for k,v in s.get('stages',{}).items():
        print(' ',k,{a:(round(b/1e9,2) if a.endswith('_ns') else b) for a,b in v.items() if a in('slot_wait_ns','reconstruct_ns','published','processed','pressure')})
P
