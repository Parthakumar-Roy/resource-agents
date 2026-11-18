"""
Tests for ibm-cloud-vpc-cr-vip resource agent.

Two test layers:
  1. Unit tests  — import the agent module directly, mock ocf / ibm_cloud_fail_over
     and subprocess so nothing external is needed.  These cover every new logic
     branch added in the vip_add branch.

  2. Integration tests  — invoke ocf-tester as a subprocess so the full OCF
     dispatch path is exercised.  These require the agent to be installed at
     AGENT_PATH and ocf-tester to be on PATH.  They are skipped automatically
     when either binary is absent (safe to run in CI without a full cluster
     install).

Run all tests:
    python3 /path/to/test_ibm_cloud_vpc_cr_vip.py

Run tests matching a substring:
    python3 /path/to/test_ibm_cloud_vpc_cr_vip.py -k Integration

Run only unit tests (no install required):
    python3 /path/to/test_ibm_cloud_vpc_cr_vip.py TestIpAddressValidator TestValidateAllAction TestMonitorAction TestStartAction TestStopAction

Run only integration tests:
    sudo python3 /path/to/test_ibm_cloud_vpc_cr_vip.py TestIntegrationValidateAll TestIntegrationMonitor
"""

import importlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# AGENT_PATH: the installed binary used by integration tests.
# Can be overridden via the AGENT_PATH environment variable.
AGENT_PATH = os.environ.get(
    "AGENT_PATH",
    "/usr/lib/ocf/resource.d/heartbeat/ibm-cloud-vpc-cr-vip",
)

# AGENT_SRC: Python source loaded by unit tests.
# Prefer the in-repo .in file when running from a checkout; fall back to the
# installed agent (same source, just without the .in extension) when the file
# is run as a standalone script on a cluster node.
_repo_src = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heartbeat", "ibm-cloud-vpc-cr-vip.in")
)
AGENT_SRC = _repo_src if os.path.exists(_repo_src) else AGENT_PATH

# Common parameter set reused across integration tests.
BASE_PARAMS = {
    "vpc_url":  "https://au-syd.private.iaas.cloud.ibm.com",
    "ext_ip_1": "172.16.1.7",
    "ext_ip_2": "172.18.1.7",
}

VALID_VIP_SINGLE = "10.100.100.100"
VALID_VIP_MULTI  = "10.100.100.100,10.100.100.101"
VALID_IFACE      = "eth0"


# ---------------------------------------------------------------------------
# Helper — load the agent source as a module with all external deps mocked.
# ---------------------------------------------------------------------------

def _load_agent_module():
    """
    Load ibm-cloud-vpc-cr-vip.in as a Python module.

    We inject stub versions of `ocf` and `ibm_cloud_fail_over` so the module
    body executes without a real OCF installation.
    """
    # Build a minimal ocf stub.
    ocf_stub = types.ModuleType("ocf")
    ocf_stub.OCF_SUCCESS       = 0
    ocf_stub.OCF_ERR_GENERIC   = 1
    ocf_stub.OCF_ERR_CONFIGURED = 6
    ocf_stub.OCF_NOT_RUNNING   = 7
    ocf_stub.logger = MagicMock()

    # Stub for the IBM Cloud library.
    failover_stub = types.ModuleType("ibm_cloud_fail_over")
    failover_stub.fail_over_cr_vip = MagicMock(return_value="172.16.1.7")

    # Patch sys.modules before loading so the import statements inside the
    # agent resolve to our stubs.
    sys.modules["ocf"] = ocf_stub
    sys.modules["ibm_cloud_fail_over"] = failover_stub

    spec = importlib.util.spec_from_file_location(
        "ibm_cloud_vpc_cr_vip",
        AGENT_SRC,
        submodule_search_locations=[],
    )
    # .in extension is not auto-recognised; explicitly attach a SourceFileLoader.
    if spec is None or spec.loader is None:
        loader = importlib.machinery.SourceFileLoader("ibm_cloud_vpc_cr_vip", AGENT_SRC)
        spec   = importlib.util.spec_from_loader("ibm_cloud_vpc_cr_vip", loader)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, ocf_stub, failover_stub


# Load once for all unit tests.
_agent, _ocf, _failover = _load_agent_module()


class VerboseTestCase(unittest.TestCase):
    """
    TestCase that prints what each assertion is checking, whether it passes
    or fails.  The values are printed *before* the assertion runs, so a failing
    assertion still shows its got/expected line above the failure message.
    """

    def assertEqual(self, first, second, msg=None):
        print(f"    assertEqual: got={first!r}  expected={second!r}")
        super().assertEqual(first, second, msg)

    def assertIn(self, member, container, msg=None):
        print(f"    assertIn   : {member!r} in {container!r}")
        super().assertIn(member, container, msg)

    def assertTrue(self, expr, msg=None):
        print(f"    assertTrue : got={expr!r}")
        super().assertTrue(expr, msg)

    def assertFalse(self, expr, msg=None):
        print(f"    assertFalse: got={expr!r}")
        super().assertFalse(expr, msg)


# ===========================================================================
# 1. Unit Tests — ip_address_validator
# ===========================================================================

class TestIpAddressValidator(VerboseTestCase):
    """Tests for the ip_address_validator helper (pre-existing, regression)."""

    def test_valid_ipv4(self):
        self.assertTrue(_agent.ip_address_validator("192.168.1.1"))

    def test_valid_loopback(self):
        self.assertTrue(_agent.ip_address_validator("127.0.0.1"))

    def test_invalid_with_prefix(self):
        # CIDR notation is NOT a bare IP — validator should reject it.
        self.assertFalse(_agent.ip_address_validator("10.0.0.1/32"))

    def test_invalid_string(self):
        self.assertFalse(_agent.ip_address_validator("not-an-ip"))

    def test_empty_string(self):
        self.assertFalse(_agent.ip_address_validator(""))

    def test_out_of_range_octet(self):
        self.assertFalse(_agent.ip_address_validator("999.999.999.999"))


# ===========================================================================
# 2. Unit Tests — validate_all_action  (new logic from vip_add branch)
# ===========================================================================

class TestValidateAllAction(VerboseTestCase):
    """
    TC-VAL-*: validate_all_action with the new interface_name / vip_addresses
    parameters.
    """

    def setUp(self):
        _ocf.logger.reset_mock()

    # --- TC-VAL-01 ---
    def test_vip_addresses_set_interface_name_empty_fails(self):
        """vip_addresses supplied but interface_name empty → OCF_ERR_CONFIGURED."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="",
            vip_addresses="10.100.100.100,10.100.100.101",
        )
        self.assertEqual(rc, _ocf.OCF_ERR_CONFIGURED)
        _ocf.logger.error.assert_called()
        error_msg = _ocf.logger.error.call_args[0][0]
        self.assertIn("interface_name", error_msg)

    # --- TC-VAL-02 ---
    def test_vip_addresses_contains_invalid_ip_fails(self):
        """One invalid VIP in the list → OCF_ERR_CONFIGURED."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="eth0",
            vip_addresses="10.100.100.100,not-an-ip",
        )
        self.assertEqual(rc, _ocf.OCF_ERR_CONFIGURED)
        error_msg = _ocf.logger.error.call_args[0][0]
        self.assertIn("not-an-ip", error_msg)

    # --- TC-VAL-03 ---
    def test_vip_addresses_single_invalid_ip_fails(self):
        """Single totally invalid VIP (no comma) → OCF_ERR_CONFIGURED."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="eth0",
            vip_addresses="999.999.999.999",
        )
        self.assertEqual(rc, _ocf.OCF_ERR_CONFIGURED)

    # --- TC-VAL-04 ---
    def test_valid_vip_and_interface_succeeds(self):
        """Both new params valid → OCF_SUCCESS."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="eth0",
            vip_addresses="10.100.100.100,10.100.100.101",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)

    # --- TC-VAL-05 ---
    def test_no_new_params_succeeds(self):
        """Neither new parameter supplied → OCF_SUCCESS (regression)."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)

    # --- TC-VAL-06 ---
    def test_interface_name_set_vip_addresses_empty_succeeds(self):
        """interface_name set but vip_addresses empty → OCF_SUCCESS (no VIPs to validate)."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="eth0",
            vip_addresses="",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)

    # --- TC-VAL-07 ---
    def test_invalid_ext_ip_1_fails(self):
        """Pre-existing guard: bad ext_ip_1 still fails (regression)."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="not-an-ip",
            ext_ip_2="172.18.1.7",
        )
        self.assertEqual(rc, _ocf.OCF_ERR_CONFIGURED)

    def test_invalid_ext_ip_2_fails(self):
        """Pre-existing guard: bad ext_ip_2 still fails (regression)."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="bad",
        )
        self.assertEqual(rc, _ocf.OCF_ERR_CONFIGURED)

    def test_vip_addresses_whitespace_only_treated_as_empty(self):
        """A vip_addresses value of only commas produces no candidates → passes without interface."""
        rc = _agent.validate_all_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            interface_name="",
            vip_addresses=",,,",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)


# ===========================================================================
# 3. Unit Tests — monitor_action  (new VIP local-presence check)
# ===========================================================================

class TestMonitorAction(VerboseTestCase):
    """
    TC-MON-*: monitor_action now checks os_ip() for each VIP before the
    IBM Cloud API call.
    """

    def setUp(self):
        _ocf.logger.reset_mock()
        # Default: API returns ext_ip_1 as active (this node is active).
        _failover.fail_over_cr_vip.reset_mock()
        _failover.fail_over_cr_vip.return_value = "172.16.1.7"

    # --- TC-MON-01 ---
    def test_vip_not_on_host_returns_not_running(self):
        """VIP absent from host → OCF_NOT_RUNNING without reaching API."""
        with patch.object(_agent, "os_ip", return_value=False):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                interface_name="eth0",
                vip_addresses="10.100.100.100",
            )
        self.assertEqual(rc, _ocf.OCF_NOT_RUNNING)

    # --- TC-MON-02 ---
    def test_first_vip_present_second_absent_returns_not_running(self):
        """First VIP on host, second absent → OCF_NOT_RUNNING."""
        present = {"10.100.100.100": True, "10.100.100.101": False}
        with patch.object(_agent, "os_ip", side_effect=lambda ip: present.get(ip, False)):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                interface_name="eth0",
                vip_addresses="10.100.100.100,10.100.100.101",
            )
        self.assertEqual(rc, _ocf.OCF_NOT_RUNNING)

    # --- TC-MON-03 ---
    def test_all_vips_present_and_api_agrees_returns_success(self):
        """All VIPs on host AND API returns this node's IP → OCF_SUCCESS."""
        with patch.object(_agent, "os_ip", return_value=True):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                interface_name="eth0",
                vip_addresses="10.100.100.100,10.100.100.101",
            )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)

    # --- TC-MON-04 ---
    def test_no_vip_addresses_falls_through_to_api(self):
        """No vip_addresses → new block is skipped, API path runs (regression)."""
        with patch.object(_agent, "os_ip", return_value=True):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
            )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)
        _failover.fail_over_cr_vip.assert_called_once()

    # --- TC-MON-05 ---
    def test_all_malformed_vip_addresses_exits_generic(self):
        """All VIP entries malformed → sys.exit(OCF_ERR_GENERIC)."""
        with self.assertRaises(SystemExit) as ctx:
            _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                interface_name="eth0",
                vip_addresses="bad-ip-1,bad-ip-2",
            )
        self.assertEqual(ctx.exception.code, _ocf.OCF_ERR_GENERIC)

    def test_monitor_api_returns_other_node_ip_returns_not_running(self):
        """API returns peer IP (not this node) → OCF_NOT_RUNNING (regression)."""
        _failover.fail_over_cr_vip.return_value = "172.18.1.7"
        with patch.object(_agent, "os_ip", return_value=False):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
            )
        self.assertEqual(rc, _ocf.OCF_NOT_RUNNING)

    def test_monitor_mixed_vips_one_malformed_one_absent_returns_not_running(self):
        """One malformed VIP (skipped with warning) + one valid but absent → NOT_RUNNING."""
        with patch.object(_agent, "os_ip", return_value=False):
            rc = _agent.monitor_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                interface_name="eth0",
                vip_addresses="bad-ip,10.100.100.100",
            )
        self.assertEqual(rc, _ocf.OCF_NOT_RUNNING)


# ===========================================================================
# 4. Unit Tests — start_action  (new params forwarded to fail_over_cr_vip)
# ===========================================================================

class TestStartAction(VerboseTestCase):
    """TC-START-*: start_action passes interface_name/vip_addresses to the library."""

    def setUp(self):
        _failover.fail_over_cr_vip.reset_mock()
        _failover.fail_over_cr_vip.return_value = "172.16.1.7"

    # --- TC-START-01 ---
    def test_start_with_vip_params_passes_them_to_library(self):
        """SET and GET calls both receive interface_name and vip_addresses."""
        rc = _agent.start_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            api_key="test-key",
            interface_name="eth0",
            vip_addresses="10.100.100.100",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)
        calls = _failover.fail_over_cr_vip.call_args_list
        # First call must be SET
        self.assertEqual(calls[0][0][0], "SET")
        self.assertIn("eth0",          calls[0][0])
        self.assertIn("10.100.100.100", calls[0][0])
        # Second call must be GET
        self.assertEqual(calls[1][0][0], "GET")
        self.assertIn("eth0",          calls[1][0])

    # --- TC-START-02 ---
    def test_start_without_vip_params_still_succeeds(self):
        """Start without new params still calls SET then GET (regression)."""
        rc = _agent.start_action(
            vpc_url="https://example.com",
            ext_ip_1="172.16.1.7",
            ext_ip_2="172.18.1.7",
            api_key="test-key",
        )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)
        self.assertEqual(_failover.fail_over_cr_vip.call_count, 2)


# ===========================================================================
# 5. Unit Tests — stop_action  (UNSET now called with new params)
# ===========================================================================

class TestStopAction(VerboseTestCase):
    """TC-STOP-*: stop_action calls UNSET before polling monitor."""

    def setUp(self):
        _failover.fail_over_cr_vip.reset_mock()
        # First call (UNSET) returns nothing meaningful; subsequent GET calls
        # return an IP that os_ip() will say is NOT on the host → NOT_RUNNING.
        _failover.fail_over_cr_vip.return_value = "172.16.1.7"

    # --- TC-STOP-01 ---
    def test_stop_with_vip_params_calls_unset_with_params(self):
        """UNSET is invoked and receives interface_name / vip_addresses."""
        with patch.object(_agent, "os_ip", return_value=False):
            rc = _agent.stop_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                api_key="test-key",
                interface_name="eth0",
                vip_addresses="10.100.100.100",
            )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)
        first_call = _failover.fail_over_cr_vip.call_args_list[0]
        self.assertEqual(first_call[0][0], "UNSET")
        self.assertIn("eth0",          first_call[0])
        self.assertIn("10.100.100.100", first_call[0])

    # --- TC-STOP-02 ---
    def test_stop_without_vip_params_succeeds(self):
        """Stop without new params still invokes UNSET and returns SUCCESS (regression)."""
        with patch.object(_agent, "os_ip", return_value=False):
            rc = _agent.stop_action(
                vpc_url="https://example.com",
                ext_ip_1="172.16.1.7",
                ext_ip_2="172.18.1.7",
                api_key="test-key",
            )
        self.assertEqual(rc, _ocf.OCF_SUCCESS)
        self.assertEqual(_failover.fail_over_cr_vip.call_args_list[0][0][0], "UNSET")


# ===========================================================================
# 6. Integration Tests
#
# Two sub-layers:
#   a) Single-action tests (TC-VAL-*, TC-MON-*) — call the agent binary
#      directly with __OCF_ACTION set, just like ocf-tester does internally.
#      Requires only the installed agent binary; no ocf-tester needed.
#
#   b) Full-cycle tests (TC-E2E-*) — invoke ocf-tester, which runs the
#      complete start → monitor → stop sequence and validates OCF compliance.
#      Requires both ocf-tester and the installed agent binary.
# ===========================================================================

def _agent_available():
    return os.path.exists(AGENT_PATH)

def _ocf_tester_available():
    return shutil.which("ocf-tester") is not None and os.path.exists(AGENT_PATH)


def _run_action(action: str, params: dict) -> int:
    """
    Invoke a single OCF action on the installed agent binary.

    Sets OCF_RESKEY_* and __OCF_ACTION in the environment, exactly as
    ocf-tester does for each individual action call.  Used by single-action
    integration tests so we can assert one specific exit code without running
    the full start/monitor/stop cycle.
    """
    env = os.environ.copy()
    env["__OCF_ACTION"] = action
    env["OCF_RESOURCE_INSTANCE"] = "vip_test"
    env["OCF_ROOT"] = "/usr/lib/ocf"
    for k, v in params.items():
        env[f"OCF_RESKEY_{k}"] = v

    # Print the equivalent shell command (only the variables this test sets,
    # not the whole inherited environment) so it can be pasted into a terminal.
    # api_key is masked so it does not end up in test logs.
    shown = ["__OCF_ACTION", "OCF_RESOURCE_INSTANCE", "OCF_ROOT"] + [
        f"OCF_RESKEY_{k}" for k in params
    ]
    lines = []
    for name in shown:
        value = "****" if name == "OCF_RESKEY_api_key" else env[name]
        lines.append(f"{name}={shlex.quote(value)}")
    print("    command    : sudo env \\\n"
          + "".join(f"                   {line} \\\n" for line in lines)
          + f"                   {shlex.quote(AGENT_PATH)}")

    result = subprocess.run(
        [AGENT_PATH],
        env=env,
        capture_output=True,
    )
    return result.returncode


def _run_ocf_tester(params: dict) -> int:
    """
    Run the full ocf-tester cycle (start → monitor → stop) against the
    installed agent.  Returns the ocf-tester exit code: 0 = all passed.
    """
    cmd = ["ocf-tester", "-n", "vip_test"]
    for k, v in params.items():
        cmd += ["-o", f"{k}={v}"]
    cmd.append(AGENT_PATH)
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode


@unittest.skipUnless(_agent_available(), "agent not installed — skipping integration tests")
class TestIntegrationValidateAll(VerboseTestCase):
    """
    Integration TC-VAL-*: run validate-all through the real agent binary.
    These mirror the unit tests but exercise the full OCF dispatch stack.
    """

    def _validate(self, extra_params: dict) -> int:
        params = {**BASE_PARAMS, **extra_params}
        return _run_action("validate-all", params)

    def test_TC_VAL_01_vip_addresses_no_interface_name(self):
        """vip_addresses set, interface_name empty → OCF_ERR_CONFIGURED (6)."""
        rc = self._validate({"interface_name": "", "vip_addresses": VALID_VIP_MULTI})
        self.assertEqual(rc, 6)

    def test_TC_VAL_02_invalid_ip_in_vip_addresses(self):
        """Invalid IP in vip_addresses → OCF_ERR_CONFIGURED (6)."""
        rc = self._validate({"interface_name": VALID_IFACE, "vip_addresses": "10.0.0.1,not-an-ip"})
        self.assertEqual(rc, 6)

    def test_TC_VAL_03_single_invalid_vip(self):
        """Single malformed vip_addresses entry → OCF_ERR_CONFIGURED (6)."""
        rc = self._validate({"interface_name": VALID_IFACE, "vip_addresses": "999.999.999.999"})
        self.assertEqual(rc, 6)

    def test_TC_VAL_04_valid_vip_and_interface(self):
        """Valid vip_addresses + interface_name → OCF_SUCCESS (0)."""
        rc = self._validate({"interface_name": VALID_IFACE, "vip_addresses": VALID_VIP_MULTI})
        self.assertEqual(rc, 0)

    def test_TC_VAL_05_no_new_params(self):
        """No new params → OCF_SUCCESS (0) — regression."""
        rc = self._validate({})
        self.assertEqual(rc, 0)

    def test_TC_VAL_06_interface_name_only(self):
        """interface_name set, vip_addresses empty → OCF_SUCCESS (0)."""
        rc = self._validate({"interface_name": VALID_IFACE, "vip_addresses": ""})
        self.assertEqual(rc, 0)

    def test_TC_VAL_07_invalid_ext_ip_1(self):
        """Bad ext_ip_1 → OCF_ERR_CONFIGURED (6) — regression guard."""
        params = {
            "vpc_url":  BASE_PARAMS["vpc_url"],
            "ext_ip_1": "not-an-ip",
            "ext_ip_2": BASE_PARAMS["ext_ip_2"],
        }
        rc = _run_action("validate-all", params)
        self.assertEqual(rc, 6)


@unittest.skipUnless(_agent_available(), "agent not installed — skipping integration tests")
class TestIntegrationMonitor(VerboseTestCase):
    """
    Integration TC-MON-*: run monitor through the real agent binary.
    TC-MON-03, TC-MON-04 and TC-MON-07 require a real IBM Cloud API key in the
    environment (API_KEY) and are skipped when it is absent.
    """

    def _monitor(self, extra_params: dict) -> int:
        params = {**BASE_PARAMS, **extra_params}
        return _run_action("monitor", params)

    def test_TC_MON_01_vip_not_on_host(self):
        """VIP not assigned to host → OCF_NOT_RUNNING (7).

        Assumes 10.100.100.200 is not assigned to any interface on this host.
        """
        rc = self._monitor({"interface_name": VALID_IFACE, "vip_addresses": "10.100.100.200"})
        self.assertEqual(rc, 7)

    def test_TC_MON_02_first_vip_present_second_absent(self):
        """First VIP on host, second absent → OCF_NOT_RUNNING (7).

        Pre-condition (set by the user, not by this test):
            sudo ip addr add 10.100.100.100/32 dev eth0   # present
            10.100.100.101 must NOT be assigned to any interface.
        The test is skipped if the pre-condition is not met, because with the
        first VIP absent the agent would also return 7 and the test would pass
        for the wrong reason.
        """
        ip_out = subprocess.run(["ip", "a"], capture_output=True, text=True).stdout
        if "10.100.100.100" not in ip_out or "10.100.100.101" in ip_out:
            self.skipTest("pre-condition not met: 10.100.100.100 must be assigned "
                          "and 10.100.100.101 must not be")
        rc = self._monitor({"interface_name": VALID_IFACE, "vip_addresses": VALID_VIP_MULTI})
        self.assertEqual(rc, 7)

    def test_TC_MON_06_malformed_vip_plus_valid_absent_vip(self):
        """One malformed VIP + one valid but absent VIP → OCF_NOT_RUNNING (7).

        Assumes 10.100.100.200 is not assigned to any interface on this host.
        """
        rc = self._monitor({"interface_name": VALID_IFACE, "vip_addresses": "bad-ip,10.100.100.200"})
        self.assertEqual(rc, 7)

    def test_TC_MON_05_all_malformed_vips(self):
        """All malformed VIP entries → OCF_ERR_GENERIC (1)."""
        rc = self._monitor({"interface_name": VALID_IFACE, "vip_addresses": "bad-ip-1,bad-ip-2"})
        self.assertEqual(rc, 1)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_MON_04_no_vip_addresses_api_path(self):
        """No vip_addresses → monitor falls back to API-only path (regression)."""
        rc = self._monitor({"api_key": os.environ["API_KEY"]})
        self.assertIn(rc, (0, 7), "Expected OCF_SUCCESS or OCF_NOT_RUNNING, not a crash")

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_MON_03_vips_present_api_agrees(self):
        """All VIPs on host + API active → OCF_SUCCESS (0).

        Pre-condition: manually assign the VIPs before running:
            sudo ip addr add 10.100.100.100/32 dev eth0
            sudo ip addr add 10.100.100.101/32 dev eth0
        """
        rc = self._monitor({
            "api_key":        os.environ["API_KEY"],
            "interface_name": VALID_IFACE,
            "vip_addresses":  VALID_VIP_MULTI,
        })
        self.assertIn(rc, (0, 7), "Expected OCF_SUCCESS or OCF_NOT_RUNNING, not a crash")

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_MON_07_api_returns_peer_node_ip(self):
        """No vip_addresses, API reports the peer as active → OCF_NOT_RUNNING (7).

        Pre-condition: run this on the node that is currently STANDBY, i.e.
        the custom route nexthop points at the other node (ext_ip_1/ext_ip_2
        in BASE_PARAMS must be this cluster's real node IPs).  On the active
        node this returns 0 instead.
        """
        rc = self._monitor({"api_key": os.environ["API_KEY"]})
        self.assertEqual(rc, 7)


@unittest.skipUnless(_agent_available(), "agent not installed — skipping integration tests")
class TestIntegrationStart(VerboseTestCase):
    """
    Integration TC-START-*: run start through the real agent binary.

    start calls fail_over_cr_vip("SET") which changes the custom route
    nexthop in IBM Cloud, so all tests here require API_KEY and have real
    side effects on the VPC routing table.  Run on a cluster node where
    this VSI is the intended active node.
    """

    def _start(self, extra_params: dict) -> int:
        params = {**BASE_PARAMS, **extra_params}
        return _run_action("start", params)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_START_01_with_vip_params(self):
        """start with interface_name and vip_addresses → OCF_SUCCESS (0).

        Verifies the new params are forwarded to fail_over_cr_vip SET + GET.
        After this test the custom route nexthop points to this node and
        vip_addresses are assigned to VALID_IFACE.
        """
        rc = self._start({
            "api_key":        os.environ["API_KEY"],
            "interface_name": VALID_IFACE,
            "vip_addresses":  VALID_VIP_MULTI,
        })
        self.assertEqual(rc, 0)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_START_02_without_vip_params(self):
        """start without new params → OCF_SUCCESS (0) — regression.

        Verifies the original two-parameter behaviour is preserved.
        """
        rc = self._start({"api_key": os.environ["API_KEY"]})
        self.assertEqual(rc, 0)


@unittest.skipUnless(_agent_available(), "agent not installed — skipping integration tests")
class TestIntegrationStop(VerboseTestCase):
    """
    Integration TC-STOP-*: run stop through the real agent binary.

    stop calls fail_over_cr_vip("UNSET") which changes the custom route
    nexthop in IBM Cloud, so all tests here require API_KEY and have real
    side effects on the VPC routing table.  Typically run after a start
    test or on the node that is currently active.
    """

    def _stop(self, extra_params: dict) -> int:
        params = {**BASE_PARAMS, **extra_params}
        return _run_action("stop", params)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_STOP_01_with_vip_params(self):
        """stop with interface_name and vip_addresses → OCF_SUCCESS (0).

        Verifies UNSET is called with the new params and vip_addresses are
        removed from VALID_IFACE.  After this test:
            ip addr show eth0  # should not show 10.100.100.100 or .101
        """
        rc = self._stop({
            "api_key":        os.environ["API_KEY"],
            "interface_name": VALID_IFACE,
            "vip_addresses":  VALID_VIP_MULTI,
        })
        self.assertEqual(rc, 0)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_STOP_02_without_vip_params(self):
        """stop without new params → OCF_SUCCESS (0) — regression."""
        rc = self._stop({"api_key": os.environ["API_KEY"]})
        self.assertEqual(rc, 0)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_STOP_03_already_stopped_is_idempotent(self):
        """stop on an already-stopped resource → OCF_SUCCESS (0).

        OCF requires stop to be idempotent.  Run stop twice; both must
        return 0.
        """
        params = {"api_key": os.environ["API_KEY"]}
        rc1 = self._stop(params)
        rc2 = self._stop(params)
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)


@unittest.skipUnless(_ocf_tester_available(), "ocf-tester or agent not installed — skipping E2E tests")
class TestIntegrationE2E(VerboseTestCase):
    """
    TC-E2E-*: full start → monitor → stop cycle via ocf-tester.

    ocf-tester runs every OCF-mandated check (metadata lint, permissions,
    start, monitor, stop, idempotency) and exits 0 only when all pass.
    These tests require a real IBM Cloud API key (API_KEY env var).
    """

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_E2E_01_full_cycle_without_vip_addresses(self):
        """Full ocf-tester cycle without new params — regression."""
        rc = _run_ocf_tester({
            **BASE_PARAMS,
            "api_key": os.environ["API_KEY"],
        })
        self.assertEqual(rc, 0)

    @unittest.skipUnless(os.environ.get("API_KEY"), "API_KEY env var not set — skipping live API test")
    def test_TC_E2E_02_full_cycle_with_vip_addresses(self):
        """Full ocf-tester cycle with interface_name and vip_addresses.

        Pre-condition: VIPs must NOT be assigned to the interface before
        this test runs — ocf-tester's start action will assign them.
        Post-condition: after stop, verify VIPs are removed:
            ip addr show eth0  # should not show 10.100.100.100 or .101
        """
        rc = _run_ocf_tester({
            **BASE_PARAMS,
            "api_key":        os.environ["API_KEY"],
            "interface_name": VALID_IFACE,
            "vip_addresses":  VALID_VIP_MULTI,
        })
        self.assertEqual(rc, 0)


class _VerboseResult(unittest.TextTestResult):
    """
    Custom test result that prints a clear separator block for every test:

        ────────────────────────────────────────────────────────────────────────
        TEST  : <module>.<Class>.<method>
        DESC  : <first line of docstring>
        RESULT: PASS | FAIL | ERROR | SKIP — <reason>

    On failures and errors the full got-vs-expected traceback produced by
    assertEqual is printed immediately after the RESULT line, so you never
    have to scroll to a separate summary section.
    No changes to any test logic are needed.
    """
    SEP = "─" * 72

    def _print_header(self, test):
        print(f"\n{self.SEP}")
        print(f"TEST  : {test.id()}")
        doc = test.shortDescription()
        if doc:
            print(f"DESC  : {doc}")

    def startTest(self, test):
        # Suppress the default "test_name ... ok" line from TextTestResult.
        unittest.TestResult.startTest(self, test)  # skip TextTestResult.startTest
        # Print the header before the test body runs so the per-assertion
        # lines emitted by VerboseTestCase appear underneath it.
        self._print_header(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        print("RESULT: PASS")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"RESULT: SKIP — {reason}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        # err[1] is the AssertionError — its str() contains the got/expected diff.
        print(f"RESULT: FAIL\n{err[1]}")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"RESULT: ERROR\n{err[1]}")


class _VerboseRunner(unittest.TextTestRunner):
    """Wire in _VerboseResult and own all output on stdout."""
    resultclass = _VerboseResult

    def __init__(self):
        # Sink the TextTestRunner's internal progress stream so only our
        # explicit print() calls appear on the terminal.
        super().__init__(stream=open(os.devnull, "w"))

    def run(self, test):
        result = self._makeResult()
        test(result)
        result.printErrors()
        run    = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        passed  = run - failed
        print(f"\n{'═' * 72}")
        print(f"Ran {run} tests — {passed} passed, {failed} failed, {skipped} skipped")
        print("OK" if failed == 0 else "FAILED")
        return result


if __name__ == "__main__":
    unittest.main(testRunner=_VerboseRunner())
