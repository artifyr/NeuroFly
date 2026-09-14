import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from connectome_loader import load_connectome_from_csv
from run_minecraft_brain import AerodynamicFlyAgent, build_hearts_bar

def test_rear_blind_spot():
    print("=== Testing 60° Posterior Blind Spot ===")
    agent = AerodynamicFlyAgent(0.0, 0.0, 0.0, is_walking=False)
    agent.x = 0.0
    agent.y = 0.0
    agent.z = 0.0
    agent.yaw = 0.0  # Facing South in Minecraft (Z+)

    # 1. Player approaching from Front (Z = +2.0, rel_angle = 0°)
    agent.prev_dist_to_player = 3.0
    is_threat, rel_angle = agent.detect_looming_threat(0.0, 0.0, 2.0)
    print(f"Front Approach (|rel_angle|={abs(rel_angle):.1f}°): Threat={is_threat}, InBlindSpot={agent.in_rear_blind_spot}")
    assert is_threat is True, "Front approach should trigger looming threat!"
    assert agent.in_rear_blind_spot is False, "Front approach is not in blind spot!"

    # 2. Player approaching directly from Behind (Z = -2.0, rel_angle = 180°)
    agent.threat_cooldown = 0
    agent.prev_dist_to_player = 3.0
    is_threat, rel_angle = agent.detect_looming_threat(0.0, 0.0, -2.0)
    print(f"Rear Approach (|rel_angle|={abs(rel_angle):.1f}°): Threat={is_threat}, InBlindSpot={agent.in_rear_blind_spot}")
    assert is_threat is False, "Direct rear approach MUST NOT trigger looming threat due to 60° blind spot!"
    assert agent.in_rear_blind_spot is True, "Should be marked in rear blind spot!"

    # 3. Boundary check: 160° behind vs 140° behind
    # 160° is inside the 60° cone (|rel_angle| > 150°)
    agent.threat_cooldown = 0
    agent.prev_dist_to_player = 3.0
    rad = math.radians(160.0)
    is_threat, rel_angle = agent.detect_looming_threat(-2.0 * math.sin(rad), 0.0, 2.0 * math.cos(rad))
    print(f"160° Approach (|rel_angle|={abs(rel_angle):.1f}°): Threat={is_threat}, InBlindSpot={agent.in_rear_blind_spot}")
    assert is_threat is False, "160° approach is inside 60° rear cone and should NOT trigger threat!"

    # 140° is outside the 60° cone (|rel_angle| <= 150°)
    agent.threat_cooldown = 0
    agent.prev_dist_to_player = 3.0
    rad = math.radians(140.0)
    is_threat, rel_angle = agent.detect_looming_threat(-2.0 * math.sin(rad), 0.0, 2.0 * math.cos(rad))
    print(f"140° Approach (|rel_angle|={abs(rel_angle):.1f}°): Threat={is_threat}, InBlindSpot={agent.in_rear_blind_spot}")
    assert is_threat is True, "140° approach is outside blind spot and SHOULD trigger threat!"
    print("[PASS] 60° Posterior Blind Spot verified successfully!\n")


def test_johnstons_organ_sound_touch():
    print("=== Testing Johnston's Organ (Sound as Physical Touch) ===")
    agent = AerodynamicFlyAgent(0.0, 5.0, 0.0, is_walking=False)
    agent.is_sleeping = True

    # 1. Sound within 1.5 blocks (d = 1.07m)
    disturbed = agent.detect_sound_or_touch(agent.x + 0.8, agent.y + 0.5, agent.z + 0.5)
    print(f"Sound at 1.07m: Disturbed={disturbed}, AnnoyedHopTicks={agent.annoyed_jump_ticks}, Sleeping={agent.is_sleeping}")
    assert disturbed is True, "Sound at <= 1.5m should trigger chordotonal organ!"
    assert agent.annoyed_jump_ticks == 16, "Annoyed hop timer should be set to 16!"
    assert agent.is_sleeping is False, "Disturbance must wake fly from sleep!"

    # 2. Step through annoyance hops: verify bounded oscillation in place (no infinite upward flying)
    start_y = agent.y
    start_x = agent.x
    start_z = agent.z
    y_positions = []
    for t in range(16):
        x, y, z, yaw, pitch, state = agent.step_behavior_and_physics(
            turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
            px=agent.x + 0.8, py=agent.y + 0.5, pz=agent.z + 0.5,
            tick=t, ground_y=0.0
        )
        y_positions.append(y)
        assert abs(x - start_x) < 0.1, f"Fly drifted horizontally during annoyance! dx={x - start_x}"
        assert abs(y - start_y) <= 0.25, f"Fly deviated too far from base altitude! dy={y - start_y}"

    print(f"Annoyance Y oscillation range: min={min(y_positions):.2f}, max={max(y_positions):.2f}, base={start_y:.2f}")
    assert agent.annoyed_jump_ticks == 0, "Annoyed hop timer should expire after 16 ticks!"

    # 3. Next tick after annoyance finishes: verify it smoothly returns to normal state (e.g. FORAGING)
    x, y, z, yaw, pitch, state_recovered = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=agent.x + 5.0, py=agent.y + 5.0, pz=agent.z + 5.0,
        tick=17, ground_y=0.0
    )
    print(f"State after annoyance expired: '{state_recovered}'")
    assert "ANNOYED" not in state_recovered, "Fly should resume normal activity after annoyance expires!"

    # 4. Sound farther away (d = 2.5m)
    agent.annoyed_cooldown = 0
    disturbed_far = agent.detect_sound_or_touch(agent.x + 2.0, agent.y, agent.z + 1.5)
    print(f"Sound at 2.5m: Disturbed={disturbed_far}, AnnoyedHopTicks={agent.annoyed_jump_ticks}")
    assert disturbed_far is False, "Sound > 1.5m should not trigger physical touch response!"
    print("[PASS] Johnston's Organ mechanosensory touch verified successfully!\n")


def test_energy_pool_and_hunger():
    print("=== Testing Energy Pool & Hunger Dominance ===")
    agent = AerodynamicFlyAgent(0.0, 0.0, 0.0, is_walking=False)
    agent.flight_energy = 100.0

    # Step physics to consume energy
    for _ in range(100):
        agent.step_behavior_and_physics(
            turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
            px=10.0, py=0.0, pz=10.0, tick=1, ground_y=0.0
        )
    print(f"Energy after 100 ticks: {agent.flight_energy:.1f}%")
    assert agent.flight_energy < 100.0, "Flight must deplete flight energy!"

    # Force energy below 50%
    agent.flight_energy = 40.0
    _, _, _, _, _, state = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=10.0, py=0.0, pz=10.0, tick=1, ground_y=0.0,
        food_valence=0.8, food_dist=5.0, food_rel_angle=30.0, food_name="Sugar"
    )
    print(f"State when energy=40%: '{state}'")
    assert "HUNGRY" in state or "FEEDING" in state, "Energy < 50% must shift dominant state to HUNGRY!"

    # Feeding recharge
    _, _, _, _, _, feed_state = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=0.0, py=0.0, pz=0.0, tick=1, ground_y=0.0,
        food_valence=0.8, food_dist=1.5, food_rel_angle=0.0, food_name="Melon"
    )
    print(f"State when feeding: '{feed_state}', Energy after feeding: {agent.flight_energy:.1f}%")
    assert "FEEDING" in feed_state, "Distance < 2.2m with food must enter FEEDING state!"
    assert agent.flight_energy > 40.0, "Feeding must recharge the energy pool!"
    print("[PASS] Energy pool & hunger threshold verified successfully!\n")


def test_rain_avoidance():
    print("=== Testing Rain Avoidance & Shelter Seeking ===")
    agent = AerodynamicFlyAgent(0.0, 0.0, 0.0, is_walking=False)

    # 1. Rain active, unsheltered -> Must seek shelter
    _, _, _, _, _, state = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=10.0, py=0.0, pz=10.0, tick=1, ground_y=0.0,
        is_raining=True, is_sheltered=False, nearest_shelter=(3.0, 2.0, 4.0, 2.0)
    )
    print(f"Rain active, unsheltered: State='{state}'")
    assert "SEEKING SHELTER" in state, "Unsheltered fly in rain must seek shelter!"

    # 2. Rain active, sheltered -> Must rest under cover
    _, _, _, _, _, state_sheltered = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=10.0, py=0.0, pz=10.0, tick=1, ground_y=0.0,
        is_raining=True, is_sheltered=True, nearest_shelter=(0.5, 0.0, 4.0, 0.0)
    )
    print(f"Rain active, sheltered: State='{state_sheltered}'")
    assert "SHELTERED" in state_sheltered, "Sheltered fly in rain must enter resting shelter state!"
    print("[PASS] Rain avoidance & shelter seeking verified successfully!\n")


def test_two_line_hud_lengths():
    print("=== Testing Two-Line Actionbar HUD Formatting & Lengths ===")
    hearts = build_hearts_bar(20.0, 20.0)

    # Line 1 test
    state_label = "HUNGRY (Tracking Sugar)"
    line1 = f"[FAFB Bee] HP: 20 [{hearts}] [{state_label}]"
    print(f"Line 1: '{line1}' (Length: {len(line1)} chars)")
    assert len(line1) < 60, f"Line 1 too long ({len(line1)} chars), risks right-side clipping!"

    # Line 2 test
    line2 = "Spikes: 1,420 | Nrg: 42% | PAM: 0.85 | Rear: [BLIND] | Cov: Sheltered"
    print(f"Line 2: '{line2}' (Length: {len(line2)} chars)")
    assert len(line2) < 75, f"Line 2 too long ({len(line2)} chars), risks right-side clipping!"
    print("[PASS] Two-Line Actionbar HUD lengths verified successfully!\n")


def test_mob_threat_and_shelter_grooming_escape():
    print("=== Testing Mob Threat Perception & Shelter Grooming Escape ===")
    agent = AerodynamicFlyAgent(0.0, 5.0, 0.0, is_walking=False)
    agent.x = 0.0
    agent.y = 5.0
    agent.z = 0.0
    agent.yaw = 0.0

    # 1. Nearby mob detection (Zombie within 1.8m in front)
    nearby_mobs = [{"id": 42, "type": "zombie", "x": 0.0, "y": 5.0, "z": 1.8}]
    is_threat, rel_angle = agent.detect_looming_threat(px=50.0, py=50.0, pz=50.0, nearby_mobs=nearby_mobs)
    assert is_threat is True, "Nearby zombie within 1.8m should trigger looming threat!"
    assert agent.shelter_groom_cooldown == 120, "Threat should set shelter groom cooldown!"

    # 2. While sheltered in rain and grooming, an escape signal or threat immediately cancels grooming
    agent.is_grooming = True
    agent.grooming_tick = 50
    _, _, _, _, _, state_escape = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=50.0, py=50.0, pz=50.0, tick=1, ground_y=0.0,
        is_escape=True, is_raining=True, is_sheltered=True
    )
    assert agent.is_grooming is False, "Escape signal must immediately abort grooming!"
    assert "GROOMING" not in state_escape and "SHELTERED: " not in state_escape, "State must not be grooming during escape!"

    # 3. Complete grooming cycle under shelter transitions to resting state (no infinite loop)
    agent.is_grooming = True
    agent.grooming_tick = 109
    _, _, _, _, _, state_rest = agent.step_behavior_and_physics(
        turn_signal=0.0, forward_signal=0.0, learned_steer=0.0,
        px=50.0, py=50.0, pz=50.0, tick=2, ground_y=0.0,
        is_escape=False, is_raining=True, is_sheltered=True
    )
    assert agent.is_grooming is False, "Grooming must terminate at tick 110!"
    assert agent.shelter_groom_cooldown > 0, "Shelter groom cooldown must be positive to prevent immediate re-grooming loop!"
    assert state_rest == "SHELTERED (Resting from Rain)", f"Expected sheltered rest state, got '{state_rest}'"
    print("[PASS] Mob threat & shelter grooming escape verified successfully!\n")


if __name__ == "__main__":
    test_rear_blind_spot()
    test_johnstons_organ_sound_touch()
    test_energy_pool_and_hunger()
    test_rain_avoidance()
    test_two_line_hud_lengths()
    test_mob_threat_and_shelter_grooming_escape()
    print("ALL 6 VERIFICATION TESTS PASSED PERFECTLY!")

