"""
Teleoperation client for sending VR commands to the manipulation environment.

This script sends position and orientation commands for:
- Root (hips)
- Head
- Left hand
- Right hand

Usage:
    python scripts/utils/teleop_client.py --host 127.0.0.1 --port 15000
"""

import socket
import struct
import time
import argparse
from typing import Tuple, List
import sys

# UDP Protocol Parameters: same packet as teleop_dummy_pub.py and the
# receivers in active_adaptation/envs/mdp/commands/{motion_tracking,teleoperation}.py:
# magic b"G6D1" + seq(u32) + 4 bodies * (pos3 + quat4) float32, little-endian.
MAGIC = b"G6D1"
MAGIC_BYTE = MAGIC  # backward-compatible name
PACK_HEADER_FMT = "<4sI"
BODY_COUNT = 4  # root, head, left_hand, right_hand


class TeleopClient:
    """Client for sending teleoperation commands via UDP."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 15000):
        """
        Initialize teleoperation client.
        
        Args:
            host: Target host IP
            port: Target UDP port
        """
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0
        
        print(f"[TeleopClient] Initialized: {host}:{port}")
    
    def send_command(
        self,
        root_pos: Tuple[float, float, float],
        root_quat: Tuple[float, float, float, float],
        head_pos: Tuple[float, float, float],
        head_quat: Tuple[float, float, float, float],
        left_hand_pos: Tuple[float, float, float],
        left_hand_quat: Tuple[float, float, float, float],
        right_hand_pos: Tuple[float, float, float],
        right_hand_quat: Tuple[float, float, float, float],
    ) -> bool:
        """
        Send teleoperation command.
        
        Args:
            root_pos: Root position (x, y, z)
            root_quat: Root quaternion (x, y, z, w)
            head_pos: Head position
            head_quat: Head quaternion
            left_hand_pos: Left hand position
            left_hand_quat: Left hand quaternion
            right_hand_pos: Right hand position
            right_hand_quat: Right hand quaternion
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Pack header
            data = struct.pack(PACK_HEADER_FMT, MAGIC, self.sequence)
            
            # Pack bodies
            bodies = [
                root_pos + root_quat,
                head_pos + head_quat,
                left_hand_pos + left_hand_quat,
                right_hand_pos + right_hand_quat,
            ]
            
            for body in bodies:
                data += struct.pack('<fffffff', *body)
            
            # Send
            self.socket.sendto(data, (self.host, self.port))
            self.sequence += 1
            
            return True
        except Exception as e:
            print(f"[TeleopClient] Error sending command: {e}")
            return False
    
    def send_idle(self) -> bool:
        """Send idle command (all zeros except quaternions at identity)."""
        return self.send_command(
            root_pos=(0.0, 0.0, 0.0),
            root_quat=(0.0, 0.0, 0.0, 1.0),
            head_pos=(0.0, 0.0, 0.0),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            left_hand_pos=(0.0, 0.0, 0.0),
            left_hand_quat=(0.0, 0.0, 0.0, 1.0),
            right_hand_pos=(0.0, 0.0, 0.0),
            right_hand_quat=(0.0, 0.0, 0.0, 1.0),
        )
    
    def close(self):
        """Close socket."""
        self.socket.close()


def demo_standing():
    """Demo: Standing pose."""
    print("\n[Demo] Standing Pose")
    client = TeleopClient()
    
    # Standing pose
    for i in range(100):
        client.send_command(
            root_pos=(0.0, 0.0, 0.0),
            root_quat=(0.0, 0.0, 0.0, 1.0),
            head_pos=(0.0, 0.0, 0.25),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            left_hand_pos=(0.15, 0.1, 0.0),
            left_hand_quat=(0.0, 0.0, 0.0, 1.0),
            right_hand_pos=(0.15, -0.1, 0.0),
            right_hand_quat=(0.0, 0.0, 0.0, 1.0),
        )
        time.sleep(0.02)
    
    client.close()


def demo_reach():
    """Demo: Reaching motion with both hands."""
    print("\n[Demo] Reaching Motion")
    client = TeleopClient()
    
    for frame in range(200):
        # Oscillate hands
        reach = 0.1 * (1.0 + 0.5 * ((frame % 100) / 100.0))
        
        client.send_command(
            root_pos=(0.0, 0.0, 0.0),
            root_quat=(0.0, 0.0, 0.0, 1.0),
            head_pos=(0.0, 0.0, 0.25),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            left_hand_pos=(reach, 0.1, 0.0),
            left_hand_quat=(0.0, 0.0, 0.0, 1.0),
            right_hand_pos=(reach, -0.1, 0.0),
            right_hand_quat=(0.0, 0.0, 0.0, 1.0),
        )
        time.sleep(0.02)
    
    client.close()


def demo_walk():
    """Demo: Walking motion."""
    print("\n[Demo] Walking Motion")
    client = TeleopClient()
    
    for frame in range(300):
        # Oscillate root position
        walk = 0.1 * (frame / 300.0)
        sway = 0.05 * (1.0 + 0.5 * ((frame % 50) / 50.0))
        
        client.send_command(
            root_pos=(walk, sway, 0.0),
            root_quat=(0.0, 0.0, 0.0, 1.0),
            head_pos=(walk, 0.0, 0.25),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            left_hand_pos=(walk + 0.15, 0.1, 0.0),
            left_hand_quat=(0.0, 0.0, 0.0, 1.0),
            right_hand_pos=(walk + 0.15, -0.1, 0.0),
            right_hand_quat=(0.0, 0.0, 0.0, 1.0),
        )
        time.sleep(0.02)
    
    client.close()


def interactive_mode(host: str, port: int):
    """Interactive mode for manual control."""
    print("\n[Interactive Mode]")
    print("Commands:")
    print("  s - Standing pose")
    print("  r - Reaching motion")
    print("  w - Walking motion")
    print("  i - Send idle command")
    print("  q - Quit")
    
    client = TeleopClient(host, port)
    
    try:
        while True:
            cmd = input("\nEnter command: ").lower().strip()
            
            if cmd == "q":
                break
            elif cmd == "s":
                # Standing
                for _ in range(10):
                    client.send_command(
                        root_pos=(0.0, 0.0, 0.0),
                        root_quat=(0.0, 0.0, 0.0, 1.0),
                        head_pos=(0.0, 0.0, 0.25),
                        head_quat=(0.0, 0.0, 0.0, 1.0),
                        left_hand_pos=(0.15, 0.1, 0.0),
                        left_hand_quat=(0.0, 0.0, 0.0, 1.0),
                        right_hand_pos=(0.15, -0.1, 0.0),
                        right_hand_quat=(0.0, 0.0, 0.0, 1.0),
                    )
                    time.sleep(0.02)
            elif cmd == "r":
                demo_reach()
            elif cmd == "w":
                demo_walk()
            elif cmd == "i":
                client.send_idle()
            else:
                print(f"Unknown command: {cmd}")
    
    finally:
        client.close()
        print("\n[TeleopClient] Closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Teleoperation client for G1 manipulation"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Target host IP (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15000,
        help="Target UDP port (default: 15000)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["stand", "reach", "walk", "interactive"],
        default="interactive",
        help="Demo mode (default: interactive)",
    )
    
    args = parser.parse_args()
    
    if args.demo == "stand":
        demo_standing()
    elif args.demo == "reach":
        demo_reach()
    elif args.demo == "walk":
        demo_walk()
    elif args.demo == "interactive":
        interactive_mode(args.host, args.port)


if __name__ == "__main__":
    main()
