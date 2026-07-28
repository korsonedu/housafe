#!/bin/bash
set -e
cd /Users/eular/Desktop/housafe/backend

# Kill old processes
pkill -f daphne 2>/dev/null || true
pkill -f "ai/placeholder" 2>/dev/null || true
sleep 1

# Clear old streams
.venv/bin/python -c "
import redis
r = redis.Redis.from_url('redis://localhost:6379/0')
r.delete('housafe:pointcloud:ingest', 'housafe:vital:ingest')
print('streams cleared')
"

# Start daphne in background
.venv/bin/daphne -p 8000 housafe.asgi:application 2>/tmp/daphne_e2e.log &
DAPHNE_PID=$!
sleep 2

if ! lsof -i :8000 -P 2>&1 | grep -q LISTEN; then
    echo "ERROR: daphne failed to start"
    cat /tmp/daphne_e2e.log
    exit 1
fi
echo "daphne running (PID=$DAPHNE_PID)"

# Start AI placeholder in background
cd /Users/eular/Desktop/housafe
/Users/eular/Desktop/housafe/backend/.venv/bin/python ai/placeholder/main.py > /tmp/ai_e2e.log 2>&1 &
AI_PID=$!
echo "AI placeholder running (PID=$AI_PID)"
sleep 1

# Run simulator - send 3 point clouds + 1 vital
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy all_proxy
export no_proxy="*"
cd /Users/eular/Desktop/housafe/backend
.venv/bin/python << 'PYEOF'
import asyncio, json, time, random
import websockets

PTS = [{'x':1,'y':0.5,'z':1.5,'velocity':0.1,'intensity':0.9} for _ in range(15)]

async def main():
    async with websockets.connect('ws://localhost:8000/ws/ingest') as ws:
        await ws.send(json.dumps({'device_id':'rad_e2e','secret':'e2e_secret'}))
        print('auth:', await ws.recv())
        for i in range(3):
            await ws.send(json.dumps({'kind':'point_cloud','payload':{'ts':int(time.time()*1000),'radar_id':'rad_e2e','room':'living','frame_id':f'f{i}','points':PTS}}))
            print(f'  pc#{i}:', await ws.recv())
            await asyncio.sleep(1.5)
        await ws.send(json.dumps({'kind':'vital','payload':{'ts':int(time.time()*1000),'radar_id':'rad_e2e','room':'living','quiet':True,'resp_rate':16.0,'heart_rate':72.0,'quality':0.9}}))
        print('  vital:', await ws.recv())
        print('simulator: all frames sent OK')
asyncio.run(main())
PYEOF

# Wait for AI to process
sleep 5

# Verify RoomState
echo ""
echo "=== Verification ==="
cd /Users/eular/Desktop/housafe/backend
.venv/bin/python << 'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ['DJANGO_SETTINGS_MODULE'] = 'housafe.settings'
import django; django.setup()
from events.models import RoomState, Alert
rs = RoomState.objects.filter(device_id='rad_e2e').first()
if rs:
    print(f'✅ RoomState found:')
    print(f'   posture={rs.posture} confidence={rs.confidence}')
    print(f'   heart_rate={rs.heart_rate} resp_rate={rs.resp_rate}')
    print(f'   moving={rs.moving} presence={rs.presence}')
else:
    print('❌ RoomState NOT FOUND')

alerts = Alert.objects.filter(device_id='rad_e2e').count()
print(f'   alerts: {alerts}')

# Check Redis stream
import redis
r = redis.Redis.from_url('redis://localhost:6379/0')
for s in r.keys('housafe:*'):
    info = r.xinfo_stream(s)
    print(f'   redis {s.decode()}: {info["length"]} msgs')
PYEOF

echo ""
echo "=== AI log ==="
cat /tmp/ai_e2e.log | tail -10

echo ""
echo "=== Daphne errors? ==="
grep -i error /tmp/daphne_e2e.log 2>/dev/null | tail -5 || echo "(none)"

# Cleanup
kill $DAPHNE_PID $AI_PID 2>/dev/null || true
echo ""
echo "=== e2e verification complete ==="
