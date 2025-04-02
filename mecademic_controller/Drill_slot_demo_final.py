import socket
import threading
import struct
import time
import numpy as np
import sys
import random
import mecademicpy.robot as mdr
import matplotlib.pyplot as plt

FORCE_SENSOR_HOST = '10.203.156.183'  # PC IP (obtained ip of PC, keep this to be same in both scripts)
FORCE_SENSOR_PORT = 12345             # Port (the port for another script is 12346)
ROBOT_IP = '192.168.0.100'      # Robot IP

DEFAULT_PLUNGE_DEPTH = 0.8
MAX_PLUNGE_DEPTH = 30
HOVER_HEIGHT = 8
CALIBRATION_TIME = 3
FORCE_THRESHOLD = 0.5
GRADIENT_THRESHOLD = 5
JOINT_VEL = 10
LINEAR_VEL = 20
LINEAR_ANGLE_VEL = 5

RESET_ALPHA = 180
RESET_BETA = 60
RESET_GAMMA = -180

class RobotForceController:
    def __init__(self):
        # Initialize force sensor data
        self.force_data = {'fx': 0, 'fy': 0, 'fz': 0,
                           'mx': 0, 'my': 0, 'mz': 0,
                           'force_grad': 0, 'torque_grad': 0}
        # Initialize robot position
        self.robot_position = {'x': 0, 'y': 0, 'z': 0}
        
        # Initialize robot connection
        self.robot = mdr.Robot()
        self.robot_connected = False
        self.robot_activated = False
        
        # Initialize force sensor socket
        self.force_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.force_thread = threading.Thread(target=self._force_listener, daemon=True)
        self.force_thread_running = False
        
        # Send robot position
        self.pos_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pos_thread = threading.Thread(target=self._position_sender, daemon=True)
        self.pos_thread_running = False
        
        # Send calibration baseline
        self.baseline_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.baseline_target = (FORCE_SENSOR_HOST, 12347)
        
        self.is_drilling = False
        self.origin = None
        
        self.pose_start_time = time.time()
        self.pose_time_list = []
        self.pose_x_list = []
        self.pose_y_list = []
        self.pose_z_list = []
        
        # New thread for pose visualization
        self.pose_viz_thread = threading.Thread(target=self.visualize_pose, daemon=True)
    
    def connect_robot(self):
        try:
            self.robot.Connect(address=ROBOT_IP)
            self.robot_connected = True
            print(f"Connected to robot at {ROBOT_IP}")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False
            
    def activate_robot(self):
        try:
            if not self.robot_connected:
                if not self.connect_robot():
                    return False
            self.robot.ActivateRobot()
            self.robot.Home()
            self.robot_activated = True
            print("Robot activated and homed")
            self.robot.SetJointVel(JOINT_VEL)
            self.robot.SetCartLinVel(LINEAR_VEL)
            self.robot.SetCartAngVel(LINEAR_ANGLE_VEL)
            return True
        except Exception as e:
            print(f"Failed to activate robot: {e}")
            return False
    
    def deactivate_robot(self):
        try:
            if self.robot_connected:
                if self.robot_activated:
                    self.robot.DeactivateRobot()
                    self.robot_activated = False
                self.robot.Disconnect()
                self.robot_connected = False
            print("Robot deactivated and disconnected")
        except Exception as e:
            print(f"Error deactivating robot: {e}")
    
    def start_force_sensor(self):
        if not self.force_thread_running:
            self.force_thread_running = True
            self.force_thread.start()
            print("Force sensor listener started")
        if not self.pos_thread_running:
            self.pos_thread_running = True
            self.pos_thread.start()
            print("Position sender thread started")
    
    def _position_sender(self):
        print("Position sender thread started")
        while self.pos_thread_running:
            try:
                if self.robot_connected and self.robot_activated:
                    pose = self.robot.GetPose()
                    self.robot_position = {'x': pose[0], 'y': pose[1], 'z': pose[2]}
                    pos_data = struct.pack('fff', pose[0], pose[1], pose[2])
                    self.pos_socket.sendto(pos_data, (FORCE_SENSOR_HOST, FORCE_SENSOR_PORT))
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in position sender: {e}")
                time.sleep(1.0)
    
    def _force_listener(self):
        local_addr = ("0.0.0.0", FORCE_SENSOR_PORT+1)
        self.force_socket.bind(local_addr)
        print(f"Listening for force sensor data on {local_addr}")
        while self.force_thread_running:
            try:
                data, addr = self.force_socket.recvfrom(1024)
                if len(data) == 32:  # 8 floats
                    fx, fy, fz, mx, my, mz, force_grad, torque_grad = struct.unpack('ffffffff', data)
                    self.force_data = {
                        'fx': fx, 'fy': fy, 'fz': fz,
                        'mx': mx, 'my': my, 'mz': mz,
                        'force_grad': force_grad, 'torque_grad': torque_grad
                    }
            except Exception as e:
                print(f"Error in force listener: {e}")
                time.sleep(0.1)
    
    def get_projected_force(self):
        return self.force_data['fx'] * np.cos(np.deg2rad(30)) - self.force_data['fz'] * np.sin(np.deg2rad(30))
    
    def get_force_gradient(self):
        return self.force_data['force_grad']
    
    def calibrate_force(self, duration=CALIBRATION_TIME):
        print(f"Calibrating force sensor for {duration} seconds...")
        start_time = time.time()
        projected_forces = []
        while time.time() - start_time < duration:
            projected_forces.append(self.get_projected_force())
            time.sleep(0.03)
        if projected_forces:
            baseline = np.mean(projected_forces)
            print(f"Calibration complete. Baseline: {baseline:.2f} N")
            try:
                self.baseline_sender.sendto(struct.pack('f', baseline), self.baseline_target)
                print("Baseline sent to force controller.")
            except Exception as e:
                print(f"Error sending baseline: {e}")
            return baseline
        else:
            print("Calibration failed.")
            return 0
    
    def move_to_pose(self, x, y, z, wait=True, timeout=5.0):
        self.robot.MovePose(x, y, z, RESET_ALPHA, RESET_BETA, RESET_GAMMA)
        if wait:
            start_time = time.time()
            target = np.array([x, y, z])
            while time.time() - start_time < timeout:
                current = np.array(self.robot.GetPose()[:3])
                if np.linalg.norm(current - target) < 0.1:
                    return True
                time.sleep(0.01)
            print("Warning: Move timeout")
            return False
        return True
    
    def move_lin(self, x, y, z, wait=True, timeout=10.0):
        self.robot.MoveLin(x, y, z, RESET_ALPHA, RESET_BETA, RESET_GAMMA)
        if wait:
            start_time = time.time()
            target = np.array([x, y, z])
            while time.time() - start_time < timeout:
                current = np.array(self.robot.GetPose()[:3])
                if np.linalg.norm(current - target) < 0.05:
                    return True
                time.sleep(0.002)
            print("Warning: Linear move timeout")
            return False
        return True
    
    def check_force_threshold(self, baseline, detection_method, force_threshold, gradient_threshold):
        if detection_method == 'force':
            current_force = self.get_projected_force()
            up_force = current_force - baseline
            print(f"Vertical up force: {up_force:.2f} N")
            if up_force > force_threshold:
                print(f"Force threshold exceeded! (up_force: {up_force:.2f} N)")
                return True
        elif detection_method == 'gradient':
            if self.get_force_gradient() > gradient_threshold:
                print(f"Gradient threshold exceeded! (gradient: {self.get_force_gradient():.2f} N/s)")
                return True
        return False
    
    def vertical_plunge_with_force_monitoring(self, x, y, target_z, baseline, detection_method, force_threshold, gradient_threshold):
        current_pos = self.robot.GetPose()
        start_z = current_pos[2]
        distance = start_z - target_z
        if distance <= 0:
            return True, True
        print(f"Plunging from Z={start_z:.2f} to Z={target_z:.2f} ({distance:.2f} mm)")
        step_size = 0.1
        steps = int(distance / step_size)
        i = 0
        while i < steps:
            next_z = max(target_z, start_z - (i+1)*step_size)
            if self.check_force_threshold(baseline, detection_method, force_threshold, gradient_threshold):
                user_input = input("Force threshold exceeded. Finish drilling? (Y/N): ").strip().upper()
                if user_input == 'Y':
                    return False, False
                else:
                    continue
            self.move_lin(x, y, next_z)
            i += 1
        return True, True
    
    def horizontal_move_with_force_monitoring(self, start_x, start_y, target_x, target_y, target_z, baseline, detection_method, force_threshold, gradient_threshold):
        dx = target_x - start_x
        dy = target_y - start_y
        total_distance = np.hypot(dx, dy)
        if total_distance <= 0:
            return True, True
        print(f"Moving horizontally from ({start_x:.2f},{start_y:.2f}) to ({target_x:.2f},{target_y:.2f})")
        step_size = 0.2
        steps = int(total_distance / step_size)
        i = 0
        while i < steps:
            fraction = (i+1) / steps
            next_x = start_x + dx * fraction
            next_y = start_y + dy * fraction
            if self.check_force_threshold(baseline, detection_method, force_threshold, gradient_threshold):
                user_input = input("Force threshold exceeded during horizontal move. Finish? (Y/N): ").strip().upper()
                if user_input == 'Y':
                    self.move_lin(target_x, target_y, target_z)
                    return False, False
                else:
                    continue
            self.move_lin(next_x, next_y, target_z)
            i += 1
        return True, True
    
    def drill_slot(self, origin, slot_left, slot_right, 
                   plunge_depth=DEFAULT_PLUNGE_DEPTH,
                   detection_method='force',
                   force_threshold=FORCE_THRESHOLD,
                   gradient_threshold=GRADIENT_THRESHOLD):
        if not self.robot_activated:
            if not self.activate_robot():
                print("Cannot drill: Robot not activated")
                return False
        
        # Save the origin position
        self.origin = [origin[0], origin[1], origin[2]]
        
        left_point = [origin[0] + slot_left[0], origin[1] + slot_left[1], origin[2] + slot_left[2]]
        right_point = [origin[0] + slot_right[0], origin[1] + slot_right[1], origin[2] + slot_right[2]]
        print(f"Drilling slot from {left_point} to {right_point}")
        hover_right = [right_point[0], right_point[1], right_point[2] + HOVER_HEIGHT]
        print(f"Moving to hover position above right endpoint: {hover_right}")
        self.move_to_pose(hover_right[0], hover_right[1], hover_right[2])
        
        baseline = self.calibrate_force()
        current_depth = 0
        is_at_right = True
        continue_drilling = True
        
        while current_depth < MAX_PLUNGE_DEPTH and continue_drilling:
            self.is_drilling = (current_depth > 1*plunge_depth)
            current_point = right_point if is_at_right else left_point
            current_x, current_y = current_point[0], current_point[1]
            current_depth += plunge_depth
            if current_depth > MAX_PLUNGE_DEPTH:
                current_depth = MAX_PLUNGE_DEPTH
            
            target_z = right_point[2] - current_depth
            print(f"Plunging at {'right' if is_at_right else 'left'} endpoint to depth {current_depth} mm (Z={target_z})")
            completed, continue_drilling = self.vertical_plunge_with_force_monitoring(
                current_x, current_y, target_z,
                baseline, detection_method, force_threshold, gradient_threshold
            )
            if not completed or not continue_drilling:
                hover_z = right_point[2] + HOVER_HEIGHT
                self.move_lin(current_x, current_y, hover_z)
                print("Drilling terminated by user choice.")
                return True
            
            next_point = left_point if is_at_right else right_point
            next_x, next_y = next_point[0], next_point[1]
            print(f"Horizontal move to ({next_x}, {next_y}, {target_z})")
            completed, continue_drilling = self.horizontal_move_with_force_monitoring(
                current_x, current_y, next_x, next_y, target_z,
                baseline, detection_method, force_threshold, gradient_threshold
            )
            if not completed or not continue_drilling:
                hover_z = right_point[2] + HOVER_HEIGHT
                self.move_lin(next_x, next_y, hover_z)
                print("Drilling terminated by user choice.")
                return True
            
            is_at_right = not is_at_right
            
            if current_depth >= MAX_PLUNGE_DEPTH and is_at_right:
                user_input = input("Maximum depth reached. Finish drilling? (Y): ").strip().upper()
                if user_input == 'Y':
                    hover_z = right_point[2] + HOVER_HEIGHT
                    self.move_lin(right_point[0], right_point[1], hover_z)
                    print("Drilling completed (max depth).")
                    return True
        
        if continue_drilling:
            current_point = right_point if is_at_right else left_point
            hover_z = right_point[2] + HOVER_HEIGHT
            self.move_lin(current_point[0], current_point[1], hover_z)
            print("Drilling completed successfully.")
        return True
    
    def visualize_pose(self):
        """
        Use matplotlib to visualize the robot's pose in real-time.
        """
        plt.ion()
        fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        
        # If origin is not set, use the robot's current pose
        if self.origin is None:
            try:
                self.origin = list(self.robot.GetPose()[:3])
            except:
                self.origin = [0, 0, 0]
        
        # Set axis limits
        ax_x.set_ylim(self.origin[0] - 100, self.origin[0] + 100)
        ax_y.set_ylim(self.origin[1] - 100, self.origin[1] + 100)
        ax_z.set_ylim(self.origin[2] - 100, self.origin[2] + 100)
        
        ax_x.set_ylabel("X (mm)")
        ax_y.set_ylabel("Y (mm)")
        ax_z.set_ylabel("Z (mm)")
        ax_z.set_xlabel("Time (s)")
        
        line_x, = ax_x.plot([], [], 'r-', label="X")
        line_y, = ax_y.plot([], [], 'g-', label="Y")
        line_z, = ax_z.plot([], [], 'b-', label="Z")
        
        ax_x.legend(loc="upper right")
        ax_y.legend(loc="upper right")
        ax_z.legend(loc="upper right")
        
        text_x = ax_x.text(0, 0, "", color="k", fontsize=10)
        text_y = ax_y.text(0, 0, "", color="k", fontsize=10)
        text_z = ax_z.text(0, 0, "", color="k", fontsize=10)
        
        plt.tight_layout()
        
        while True:
            try:
                now = time.time() - self.pose_start_time
                self.pose_time_list.append(now)
                self.pose_x_list.append(self.robot_position['x'])
                self.pose_y_list.append(self.robot_position['y'])
                self.pose_z_list.append(self.robot_position['z'])
                
                # Only keep the last 10 seconds of data
                while self.pose_time_list and (now - self.pose_time_list[0]) > 10.0:
                    self.pose_time_list.pop(0)
                    self.pose_x_list.pop(0)
                    self.pose_y_list.pop(0)
                    self.pose_z_list.pop(0)
                
                line_x.set_data(self.pose_time_list, self.pose_x_list)
                line_y.set_data(self.pose_time_list, self.pose_y_list)
                line_z.set_data(self.pose_time_list, self.pose_z_list)
  
                if len(self.pose_time_list) > 1:
                    t_min = self.pose_time_list[0]
                    t_max = self.pose_time_list[-1]
                    ax_x.set_xlim(t_min, t_max)
                    ax_y.set_xlim(t_min, t_max)
                    ax_z.set_xlim(t_min, t_max)
                
                if self.pose_time_list:
                    last_t = self.pose_time_list[-1]
                    last_x = self.pose_x_list[-1]
                    last_y = self.pose_y_list[-1]
                    last_z = self.pose_z_list[-1]
                    
                    text_x.set_position((last_t-1, last_x-1.0))
                    text_x.set_text(f"{last_x:.3f}")
                    text_y.set_position((last_t-1, last_y-1.0))
                    text_y.set_text(f"{last_y:.3f}")
                    text_z.set_position((last_t-1, last_z-1.0))
                    text_z.set_text(f"{last_z:.3f}")
                
                plt.pause(0.1)
            except Exception as e:
                print(f"Pose visualization error: {e}")
                time.sleep(0.1)
    
    def shutdown(self):
        self.force_thread_running = False
        if self.force_thread.is_alive():
            self.force_thread.join(timeout=1.0)
        self.pos_thread_running = False
        if self.pos_thread.is_alive():
            self.pos_thread.join(timeout=1.0)
        self.deactivate_robot()
        print("Controller shutdown complete")


def main():
    controller = RobotForceController()
    
    # Start force sensor listener
    controller.start_force_sensor()
    
    # Start pose visualization thread
    controller.pose_viz_thread.start()
    
    # Connect and activate the robot
    if not controller.connect_robot():
        print("Failed to connect to robot, exiting...")
        return
    if not controller.activate_robot():
        print("Failed to activate robot, exiting...")
        controller.deactivate_robot()
        return
    
    try:
        while True:
            print("\nRobot Force Controller Commands:")
            print("1. Drill Slot")
            print("2. Move to Position")
            print("3. Exit")
            choice = input("Enter choice (1-3): ")
            if choice == '1':
                print("\nUsing default origin and slot endpoints.") # This value should be the origin of the skull.
                ox, oy, oz = 187.63, 13.694, 111.855
                print("\nEnter slot left endpoint (relative to origin):") # This value should be the left endpoint of the slot (wrt origin).
                lx = float(input("Left X (mm): "))
                ly = float(input("Left Y (mm): "))
                lz = float(input("Left Z (mm): "))
                
                print("\nEnter slot right endpoint (relative to origin):") # This value should be the right endpoint of the slot (wrt origin).
                rx = float(input("Right X (mm): "))
                ry = float(input("Right Y (mm): "))
                rz = float(input("Right Z (mm): "))
                plunge = float(input(f"Plunge depth per pass (mm) [default: {DEFAULT_PLUNGE_DEPTH}]: ") or DEFAULT_PLUNGE_DEPTH) # This value should be the plunge depth per pass.
                detection = input("Detection method (force/gradient) [default: force]: ").lower() or "force" # Define the detection method.
                controller.drill_slot(
                    [ox, oy, oz],
                    [lx, ly, lz],
                    [rx, ry, rz],
                    plunge_depth=plunge,
                    detection_method=detection
                )
            elif choice == '2':
                try:
                    x = float(input("X position (mm): ") or "187.63")
                    y = float(input("Y position (mm): ") or "13.694")
                    z = float(input("Z position (mm): ") or "111.855")
                    controller.move_to_pose(x, y, z)
                    print(f"Moving to position: [{x}, {y}, {z}]")
                except Exception as e:
                    print(f"Invalid input: {e}")
            elif choice == '3':
                break
            else:
                print("Invalid choice, please try again.")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    finally:
        controller.shutdown()
        print("Program terminated.")


if __name__ == "__main__":
    main()
