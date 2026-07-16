"""最小灌数脚本：python feed.py <device_id> <secret> [ws_url]"""
import asyncio, sys, time, json, websockets

async def main(device_id, secret, url="ws://localhost:8000/ws/ingest"):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"device_id":device_id,"secret":secret}))
        print(await ws.recv())
        seq = 0
        script = [("presence",{"presence":True,"moving":False}),
                  ("posture",{"posture":"stand","confidence":0.95}),
                  ("posture",{"posture":"walk","confidence":0.9}),
                  ("vital",{"quiet":True,"resp_rate":16.0,"heart_rate":72.0,"quality":0.8}),
                  ("posture",{"posture":"lie","confidence":0.92})]
        for kind, fields in script:
            seq += 1
            payload = {"ts":int(time.time()*1000),"radar_id":device_id,"room":"bedroom","seq":seq, **fields}
            await ws.send(json.dumps({"kind":kind,"payload":payload}))
            print(await ws.recv()); await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
