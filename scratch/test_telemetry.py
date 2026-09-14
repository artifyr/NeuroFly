import asyncio
import json
import websockets

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        init_msg = await ws.recv()
        print('Init received:', init_msg[:80] + '...')

        # Send agent telemetry packet
        agent_pkt = {
            'type': 'agent_telemetry',
            'source': 'Minecraft',
            'entity': 'bee',
            'pos': [-38.2, -58.0, 175.4],
            'yaw': 45.0,
            'pitch': -2.1,
            'hp': 20.0,
            'max_hp': 20.0,
            'state': 'FEEDING (Flower) [+HP]',
            'speed': 2.4,
            'nearest_light': {'dist': 2.4, 'name': 'Torch'},
            'nearest_hazard': {'dist': 8.0, 'name': 'Campfire'},
            'nearest_food': {'dist': 1.2, 'name': 'Flower (Sugar)'},
            'player_dist': 4.5,
            'is_night': False
        }
        await ws.send(json.dumps(agent_pkt))
        print('Agent telemetry sent!')

        # Receive next telemetry broadcast
        for _ in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get('type') == 'telemetry':
                step = data.get('step')
                count = data.get('count')
                da = data.get('dopamine')
                print(f'Telemetry step {step}: spikes={count}, DA={da}')
                agent = data.get('agent')
                print('Broadcasted Agent info:', agent)
                assert agent is not None, 'Agent should be present in telemetry broadcast!'
                assert agent['connected'] is True, f"Expected connected=True, got {agent.get('connected')}"
                assert agent['state'] == 'FEEDING (Flower) [+HP]'
                assert agent['pos'] == [-38.2, -58.0, 175.4]
                print('VERIFICATION SUCCESSFUL: Minecraft Fly Brain Telemetry is live in WebSocket stream!')
                return

    raise RuntimeError("Failed to receive telemetry")

if __name__ == '__main__':
    asyncio.run(test())
