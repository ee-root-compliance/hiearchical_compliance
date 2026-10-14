"""
Test script for teleoperation system.

This script verifies that the teleoperation system components are working correctly.

Usage:
    python scripts/utils/test_teleop.py
"""

import socket
import struct
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("\n[Test 1/5] Testing imports...")
    
    try:
        from active_adaptation.envs.manipulation import ManipulationEnv
        print("  ✅ ManipulationEnv imported")
    except Exception as e:
        print(f"  ❌ Failed to import ManipulationEnv: {e}")
        return False
    
    try:
        from active_adaptation.envs.mdp.commands.teleoperation import TeleopCommand
        print("  ✅ TeleopCommand imported")
    except Exception as e:
        print(f"  ❌ Failed to import TeleopCommand: {e}")
        return False
    
    try:
        from scripts.utils.teleop_client import TeleopClient
        print("  ✅ TeleopClient imported")
    except Exception as e:
        print(f"  ❌ Failed to import TeleopClient: {e}")
        return False
    
    return True


def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\n[Test 2/5] Testing config loading...")
    
    try:
        from omegaconf import OmegaConf
        
        # Test manipulation config
        cfg = OmegaConf.load("cfg/manipulation.yaml")
        print(f"  ✅ Loaded cfg/manipulation.yaml")
        
        # Test task config
        task_cfg = OmegaConf.load("cfg/task/G1/G1_manipulation.yaml")
        print(f"  ✅ Loaded cfg/task/G1/G1_manipulation.yaml")
        
        # Verify command type
        if "command" in task_cfg:
            print(f"  ✅ Found command config: {task_cfg.command._target_}")
        else:
            print(f"  ⚠️  No command config found in task")
        
        return True
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False


def test_udp_socket():
    """Test UDP socket creation and send/receive."""
    print("\n[Test 3/5] Testing UDP socket...")
    
    try:
        # Create sender socket
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("  ✅ Created sender socket")
        
        # Create receiver socket
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))  # Bind to any available port
        port = receiver.getsockname()[1]
        print(f"  ✅ Created receiver socket on port {port}")
        
        # Set timeout for receive
        receiver.settimeout(2.0)
        
        # Send test packet
        test_data = b"TEST"
        sender.sendto(test_data, ("127.0.0.1", port))
        print("  ✅ Sent test packet")
        
        # Receive test packet
        data, addr = receiver.recvfrom(1024)
        if data == test_data:
            print("  ✅ Received test packet correctly")
        else:
            print(f"  ❌ Received incorrect data: {data}")
            return False
        
        sender.close()
        receiver.close()
        print("  ✅ Closed sockets")
        
        return True
    except Exception as e:
        print(f"  ❌ UDP socket test failed: {e}")
        return False


def test_teleop_packet():
    """Test teleoperation packet format."""
    print("\n[Test 4/5] Testing teleoperation packet format...")
    
    try:
        from scripts.utils.teleop_client import TeleopClient, MAGIC, PACK_HEADER_FMT

        # Create packet
        client = TeleopClient()
        print("  ✅ Created TeleopClient")

        # Verify packet format: G6D1 header + 4 bodies * (pos3 + quat4)
        seq = 0
        data = struct.pack(PACK_HEADER_FMT, MAGIC, seq)
        for i in range(4):
            body = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            data += struct.pack('<7f', *body)

        expected_size = 4 + 4 + 4 * 28  # 120 bytes
        if len(data) == expected_size:
            print(f"  ✅ Packet size correct: {len(data)} bytes")
        else:
            print(f"  ❌ Packet size incorrect: {len(data)} != {expected_size}")
            return False
        if data[:4] != MAGIC:
            print(f"  ❌ Packet magic incorrect: {data[:4]!r} != {MAGIC!r}")
            return False
        
        client.close()
        return True
    except Exception as e:
        print(f"  ❌ Teleoperation packet test failed: {e}")
        return False


def test_teleop_client():
    """Test TeleopClient send/receive."""
    print("\n[Test 5/5] Testing TeleopClient send/receive...")
    
    try:
        from scripts.utils.teleop_client import TeleopClient
        import threading
        
        # Start receiver on port 15001
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 15001))
        receiver.settimeout(2.0)
        print("  ✅ Started test receiver on port 15001")
        
        # Create client pointing to receiver
        client = TeleopClient(host="127.0.0.1", port=15001)
        print("  ✅ Created client")
        
        # Send command
        success = client.send_command(
            root_pos=(0.0, 0.0, 0.0),
            root_quat=(0.0, 0.0, 0.0, 1.0),
            head_pos=(0.0, 0.0, 0.25),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            left_hand_pos=(0.15, 0.1, 0.0),
            left_hand_quat=(0.0, 0.0, 0.0, 1.0),
            right_hand_pos=(0.15, -0.1, 0.0),
            right_hand_quat=(0.0, 0.0, 0.0, 1.0),
        )
        
        if success:
            print("  ✅ Sent command")
        else:
            print("  ❌ Failed to send command")
            return False
        
        # Receive command
        try:
            data, addr = receiver.recvfrom(1024)
            print(f"  ✅ Received command ({len(data)} bytes)")
        except socket.timeout:
            print("  ❌ Timeout receiving command")
            return False
        
        client.close()
        receiver.close()
        return True
    except Exception as e:
        print(f"  ❌ TeleopClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TELEOPERATION SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("UDP Socket", test_udp_socket()))
    results.append(("Teleop Packet", test_teleop_packet()))
    results.append(("TeleopClient", test_teleop_client()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Teleoperation system is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
