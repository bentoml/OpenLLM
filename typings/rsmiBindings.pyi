# See https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/python_smi_tools/rsmiBindings.py
import ctypes
from typing import LiteralString


class rocmsmi(ctypes.CDLL):
    @staticmethod
    def rsmi_num_monitor_devices(num_devices: ctypes._CArgObject) -> LiteralString: ...

# Device ID
dv_id: ctypes.c_uint64 = ...
# GPU ID
gpu_id: ctypes.c_uint32 = ...

# Policy enums
RSMI_MAX_NUM_FREQUENCIES = 32
RSMI_MAX_FAN_SPEED = 255
RSMI_NUM_VOLTAGE_CURVE_POINTS = 3

class rsmi_status_t(ctypes.c_int):
    RSMI_STATUS_SUCCESS = 0x0
    RSMI_STATUS_INVALID_ARGS = 0x1
    RSMI_STATUS_NOT_SUPPORTED = 0x2
    RSMI_STATUS_FILE_ERROR = 0x3
    RSMI_STATUS_PERMISSION = 0x4
    RSMI_STATUS_OUT_OF_RESOURCES = 0x5
    RSMI_STATUS_INTERNAL_EXCEPTION = 0x6
    RSMI_STATUS_INPUT_OUT_OF_BOUNDS = 0x7
    RSMI_STATUS_INIT_ERROR = 0x8
    RSMI_INITIALIZATION_ERROR = RSMI_STATUS_INIT_ERROR
    RSMI_STATUS_NOT_YET_IMPLEMENTED = 0x9
    RSMI_STATUS_NOT_FOUND = 0xA
    RSMI_STATUS_INSUFFICIENT_SIZE = 0xB
    RSMI_STATUS_INTERRUPT = 0xC
    RSMI_STATUS_UNEXPECTED_SIZE = 0xD
    RSMI_STATUS_NO_DATA = 0xE
    RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF

# Dictionary of rsmi ret codes and it's verbose output
rsmi_status_verbose_err_out = {
    rsmi_status_t.RSMI_STATUS_SUCCESS: "Operation was successful",
    rsmi_status_t.RSMI_STATUS_INVALID_ARGS: "Invalid arguments provided",
    rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED: "Not supported on the given system",
    rsmi_status_t.RSMI_STATUS_FILE_ERROR: "Problem accessing a file",
    rsmi_status_t.RSMI_STATUS_PERMISSION: "Permission denied",
    rsmi_status_t.RSMI_STATUS_OUT_OF_RESOURCES: "Unable to acquire memory or other resource",
    rsmi_status_t.RSMI_STATUS_INTERNAL_EXCEPTION: "An internal exception was caught",
    rsmi_status_t.RSMI_STATUS_INPUT_OUT_OF_BOUNDS: "Provided input is out of allowable or safe range",
    rsmi_status_t.RSMI_INITIALIZATION_ERROR: "Error occured during rsmi initialization",
    rsmi_status_t.RSMI_STATUS_NOT_YET_IMPLEMENTED: "Requested function is not implemented on this setup",
    rsmi_status_t.RSMI_STATUS_NOT_FOUND: "Item searched for but not found",
    rsmi_status_t.RSMI_STATUS_INSUFFICIENT_SIZE: "Insufficient resources available",
    rsmi_status_t.RSMI_STATUS_INTERRUPT: "Interrupt occured during execution",
    rsmi_status_t.RSMI_STATUS_UNEXPECTED_SIZE: "Unexpected amount of data read",
    rsmi_status_t.RSMI_STATUS_NO_DATA: "No data found for the given input",
    rsmi_status_t.RSMI_STATUS_UNKNOWN_ERROR: "Unknown error occured",
}
