import socket
import struct
import time
import threading
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime
import pysoem
import ctypes
from collections import namedtuple

class ForceSensorBridge:
    """
    Bridge class that reads from Bota force sensor via EtherCAT and visualizes force data.
    It visualizes 3-axis force (Fx, Fy, Fz), the vertical force (up_force = projected_force - baseline),
    and the smoothed up_force gradient. The gradient smoothing window can be adjusted via the
    'gradient_smoothing_window' variable.
    """
    BOTA_VENDOR_ID = 0xB07A
    BOTA_PRODUCT_CODE = 0x00000001
    SINC_LENGTH = 256
    time_step = 0.5
    
    def __init__(self, ifname, robot_port=12346, robot_ip='10.203.156.183'):
        self._ifname = ifname
        self.robot_port = robot_port
        self.robot_ip = robot_ip
        
        # EtherCAT master setup
        self._pd_thread_stop_event = threading.Event()
        self._ch_thread_stop_event = threading.Event()
        self._actual_wkc = 0
        self._master = pysoem.Master()
        self._master.in_op = False
        self._master.do_check_state = False
        SlaveSet = namedtuple('SlaveSet', 'name product_code config_func')
        self._expected_slave_layout = {0: SlaveSet('BFT-MEDS-ECAT-M8', self.BOTA_PRODUCT_CODE, self.bota_sensor_setup)}
        
        # Socket to forward data (if needed)
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Data arrays for visualization (fixed length)
        self.data_points = 100
        self.time_data = np.linspace(0, 10, self.data_points)
        self.fx_data = np.zeros(self.data_points)
        self.fy_data = np.zeros(self.data_points)
        self.fz_data = np.zeros(self.data_points)
        self.up_force_data = np.zeros(self.data_points)  # vertical force = projected force - baseline
        self.up_force_grad_data = np.zeros(self.data_points)  # smoothed up force gradient
        
        # Parameters for up force gradient smoothing:
        self.gradient_smoothing_window = 10  # Adjustable window size for smoothing
        self.up_force_buffer = np.zeros(self.gradient_smoothing_window)
        self.up_force_buffer_count = 0
        
        # Baseline force, received from drill program via UDP; initial value 0
        self.baseline_force = 0.0
        self.baseline_port = 12347
        self.baseline_receiver_thread = threading.Thread(target=self._baseline_receiver, daemon=True)
        
        # Data storage for saving
        self.save_data = []
        self.should_save = False
        self.start_time = None
        
        print("Force Sensor Bridge initialized")
        print(f"Using EtherCAT interface: {self._ifname}")
        print("Force visualization active")
    
    def bota_sensor_setup(self, slave_pos):
        print("Setting up Bota force-torque sensor...")
        slave = self._master.slaves[slave_pos]
        try:
            slave.sdo_write(0x8010, 1, bytes(ctypes.c_uint8(1)))
            slave.sdo_write(0x8010, 2, bytes(ctypes.c_uint8(0)))
            slave.sdo_write(0x8010, 3, bytes(ctypes.c_uint8(1)))
            slave.sdo_write(0x8006, 2, bytes(ctypes.c_uint8(1)))
            slave.sdo_write(0x8006, 3, bytes(ctypes.c_uint8(0)))
            slave.sdo_write(0x8006, 4, bytes(ctypes.c_uint8(0)))
            slave.sdo_write(0x8006, 1, bytes(ctypes.c_uint16(self.SINC_LENGTH)))
            sampling_rate = struct.unpack('h', slave.sdo_read(0x8011, 0))[0]
            print(f"Sampling rate {sampling_rate}")
            if sampling_rate > 0:
                self.time_step = 1.0 / float(sampling_rate)
            print(f"time step {self.time_step}")
        except Exception as e:
            print(f"Error in bota_sensor_setup: {e}")
            raise
    
    def _baseline_receiver(self):
        """Receive baseline force from drill program via UDP on port self.baseline_port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.baseline_port))
        print(f"Baseline receiver started on port {self.baseline_port}")
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                if len(data) >= 4:
                    self.baseline_force = struct.unpack('f', data[:4])[0]
                    print(f"Received new baseline force: {self.baseline_force:.2f} N")
            except Exception as e:
                print(f"Baseline receiver error: {e}")
                time.sleep(0.1)
    
    def run(self):
        save_response = input("Do you want to save sensor data? (Y/N): ").strip().upper()
        self.should_save = (save_response == 'Y')
        if self.should_save:
            self.start_time = time.time()
            print("Data recording started")
        self.baseline_receiver_thread.start()
        try:
            print("Opening EtherCAT interface...")
            self._master.open(self._ifname)
            if not self._master.config_init() > 0:
                self._master.close()
                raise RuntimeError('No slave found')
            slave_count = len(self._master.slaves)
            print(f"Found {slave_count} EtherCAT slaves")
            for i, slave in enumerate(self._master.slaves):
                print(f"Slave {i}: Vendor ID=0x{slave.man:08X}, Product=0x{slave.id:08X}")
                if not ((slave.man == self.BOTA_VENDOR_ID) and (slave.id == self._expected_slave_layout[i].product_code)):
                    self._master.close()
                    raise RuntimeError('Unexpected slave layout')
                slave.config_func = self._expected_slave_layout[i].config_func
                slave.is_lost = False
            print("Configuring PDO mapping...")
            self._master.config_map()
            print("Waiting for SAFEOP state...")
            if self._master.state_check(pysoem.SAFEOP_STATE, 50000) != pysoem.SAFEOP_STATE:
                self._master.close()
                raise RuntimeError('Not all slaves reached SAFEOP state')
            print("Requesting OP state...")
            self._master.state = pysoem.OP_STATE
            check_thread = threading.Thread(target=self._check_thread, daemon=True)
            check_thread.start()
            proc_thread = threading.Thread(target=self._processdata_thread, daemon=True)
            proc_thread.start()
            
            self._master.send_processdata()
            self._master.receive_processdata(2000)
            self._master.write_state()
            
            all_slaves_reached_op_state = False
            print("Waiting for all slaves to reach OP state...")
            for i in range(40):
                self._master.state_check(pysoem.OP_STATE, 50000)
                if self._master.state == pysoem.OP_STATE:
                    all_slaves_reached_op_state = True
                    print("All slaves reached OP state!")
                    break
                print(f"Try {i+1}/40: Not all slaves in OP state yet")
                time.sleep(0.1)
            
            if all_slaves_reached_op_state:
                self._visualize_data()
            else:
                print("Failed to reach OP state for all slaves")
            
            self._pd_thread_stop_event.set()
            self._ch_thread_stop_event.set()
            proc_thread.join()
            check_thread.join()
            self._master.state = pysoem.INIT_STATE
            self._master.write_state()
            self._master.close()
            
            if self.should_save:
                self.save_recorded_data()
            
            if not all_slaves_reached_op_state:
                raise RuntimeError('Not all slaves reached OP state')
                
        except Exception as e:
            print(f"Error in Force Sensor Bridge: {e}")
            try:
                self._master.close()
            except:
                pass
            sys.exit(1)
    
    def _processdata_thread(self):
        print("Process data thread started")
        while not self._pd_thread_stop_event.is_set():
            try:
                start_time = time.perf_counter()
                self._master.send_processdata()
                self._actual_wkc = self._master.receive_processdata(2000)
                if self._actual_wkc != self._master.expected_wkc:
                    self._sleep(self.time_step)
                    continue
                sensor_input = self._master.slaves[0].input
                fx = struct.unpack_from('f', sensor_input, 5)[0]
                fy = struct.unpack_from('f', sensor_input, 9)[0]
                fz = struct.unpack_from('f', sensor_input, 13)[0]
                
                self.fx_data = np.roll(self.fx_data, -1)
                self.fy_data = np.roll(self.fy_data, -1)
                self.fz_data = np.roll(self.fz_data, -1)
                self.fx_data[-1] = fx
                self.fy_data[-1] = fy
                self.fz_data[-1] = fz
                
                # Calculate force in vertical direction
                proj_force = fx * np.cos(np.deg2rad(30)) - fz * np.sin(np.deg2rad(30))
                up_force = proj_force - self.baseline_force
                self.up_force_data = np.roll(self.up_force_data, -1)
                self.up_force_data[-1] = up_force
                
                # Update up_force gradient buffer
                if self.up_force_buffer_count < self.gradient_smoothing_window:
                    self.up_force_buffer[self.up_force_buffer_count] = up_force
                    self.up_force_buffer_count += 1
                else:
                    self.up_force_buffer = np.roll(self.up_force_buffer, -1)
                    self.up_force_buffer[-1] = up_force
                
                if self.up_force_buffer_count == self.gradient_smoothing_window:
                    half = self.gradient_smoothing_window // 2
                    avg_first = np.mean(self.up_force_buffer[:half])
                    avg_second = np.mean(self.up_force_buffer[half:])
                    up_force_grad = (avg_second - avg_first) / (half * self.time_step)
                else:
                    up_force_grad = 0.0
                
                self.up_force_grad_data = np.roll(self.up_force_grad_data, -1)
                self.up_force_grad_data[-1] = up_force_grad
                
                data_to_send = struct.pack('ffffffff', fx, fy, fz, 0.0, 0.0, 0.0, 0.0, 0.0)
                try:
                    self.robot_socket.sendto(data_to_send, (self.robot_ip, self.robot_port))
                except Exception:
                    pass
                
                if self.should_save:
                    if self.start_time is None:
                        self.start_time = time.time()
                    t_current = time.time() - self.start_time
                    self.save_data.append([t_current, fx, fy, fz, up_force, up_force_grad])
                
                time_diff = time.perf_counter() - start_time
                if time_diff < self.time_step:
                    self._sleep(self.time_step - time_diff)
            except Exception as e:
                print(f"Error in process data thread: {e}")
                time.sleep(0.1)
    
    def _check_thread(self):
        print("State check thread started")
        while not self._ch_thread_stop_event.is_set():
            try:
                if self._master.in_op and ((self._actual_wkc < self._master.expected_wkc) or self._master.do_check_state):
                    self._master.do_check_state = False
                    self._master.read_state()
                    for i, slave in enumerate(self._master.slaves):
                        if slave.state != pysoem.OP_STATE:
                            self._master.do_check_state = True
                            self._check_slave(slave, i)
                    if not self._master.do_check_state:
                        print("OK: all slaves operational.")
            except Exception as e:
                print(f"Error in check thread: {e}")
            time.sleep(self.time_step)
    
    @staticmethod
    def _check_slave(slave, pos):
        if slave.state == (pysoem.SAFEOP_STATE + pysoem.STATE_ERROR):
            print(f"ERROR: slave {pos} in SAFE_OP+ERROR, attempting ack.")
            slave.state = pysoem.SAFEOP_STATE + pysoem.STATE_ACK
            slave.write_state()
        elif slave.state == pysoem.SAFEOP_STATE:
            print(f"WARNING: slave {pos} in SAFE_OP, trying to change to OPERATIONAL.")
            slave.state = pysoem.OP_STATE
            slave.write_state()
        elif slave.state > pysoem.NONE_STATE:
            if slave.reconfig():
                slave.is_lost = False
                print(f"MESSAGE: slave {pos} reconfigured")
        elif not slave.is_lost:
            slave.state_check(pysoem.OP_STATE)
            if slave.state == pysoem.NONE_STATE:
                slave.is_lost = True
                print(f"ERROR: slave {pos} lost")
        if slave.is_lost:
            if slave.state == pysoem.NONE_STATE:
                if slave.recover():
                    slave.is_lost = False
                    print(f"MESSAGE: slave {pos} recovered")
            else:
                slave.is_lost = False
            print(f"MESSAGE: slave {pos} found")
    
    @staticmethod
    def _sleep(duration, get_now=time.perf_counter):
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()
    
    def _visualize_data(self):
        """
        Create a visualization figure with three subplots:
          - Subplot 1: Displays the Fx, Fy, Fz curves;
          - Subplot 2: Displays the vertical force (up_force) curve;
          - Subplot 3: Displays the gradient of up_force (calculated using a smoothing algorithm).
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Subplot1: 3-axis force (Fx, Fy, Fz)
        line_fx, = ax1.plot(self.time_data, self.fx_data, label="Fx")
        line_fy, = ax1.plot(self.time_data, self.fy_data, label="Fy")
        line_fz, = ax1.plot(self.time_data, self.fz_data, label="Fz")
        ax1.set_ylabel("Force (N)")
        ax1.set_title("3-Axis Force")
        ax1.legend()
        ax1.set_ylim(-30, 30)
        
        # Subplot2: Vertical force (up_force)
        line_up, = ax2.plot(self.time_data, self.up_force_data, label="Vertical Force (up_force)", color="m")
        ax2.set_ylabel("Vertical Force (N)")
        ax2.set_title("Calibrated Vertical Force (up_force)")
        ax2.legend()
        ax2.set_ylim(-20, 20)
        
        # Subplot3: Up force gradient
        line_up_grad, = ax3.plot(self.time_data, self.up_force_grad_data, label="Up Force Gradient", color="c")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Up Force Gradient (N/s)")
        ax3.set_title("Up Force Gradient (smoothed)")
        ax3.legend()
        ax3.set_ylim(-20, 20)
        
        plt.tight_layout()
        
        def update(frame):
            line_fx.set_ydata(self.fx_data)
            line_fy.set_ydata(self.fy_data)
            line_fz.set_ydata(self.fz_data)
            line_up.set_ydata(self.up_force_data)
            line_up_grad.set_ydata(self.up_force_grad_data)
            return (line_fx, line_fy, line_fz, line_up, line_up_grad)
        
        ani = FuncAnimation(fig, update, interval=100, blit=True)
        plt.show()
    
    def save_recorded_data(self):
        if not self.save_data:
            print("No data to save")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"force_data_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Fx", "Fy", "Fz", "UpForce", "UpForceGrad"])
            writer.writerows(self.save_data)
        print(f"Force data saved to {filename}")

def main():
    print("Force Sensor Bridge started")
    adapter_name = "\\Device\\NPF_{B6CC7F7F-DD67-44DF-AEED-52A433F83B5D}" # The value depends on the device
    robot_port = 12346  # Port (the port for another script is 12345)
    robot_ip = "10.203.156.183" # PC IP (obtained ip of PC, keep this to be same in both scripts)
    if len(sys.argv) > 1:
        adapter_name = sys.argv[1]
    if len(sys.argv) > 2:
        robot_ip = sys.argv[2]
    if len(sys.argv) > 3:
        robot_port = int(sys.argv[3])
    try:
        bridge = ForceSensorBridge(adapter_name, robot_port, robot_ip)
        bridge.run()
    except KeyboardInterrupt:
        print("Force Sensor Bridge terminated by user")
    except Exception as e:
        print(f"Force Sensor Bridge failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
