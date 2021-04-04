# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py
# and https://github.com/qulacs/cirq-qulacs/blob/master/cirqqulacs/qulacs_simulator.py
#
# Adapted by Daniel Strano.
# Many thanks to the qulacs team for providing an open source example of a Cirq provider.
# Many thanks to Adam Kelley for an example of a third-party Qiskit provider.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name


import numpy as np
from collections import Counter
import logging
import os
from typing import Dict
from .qrack_controller_wrapper import qrack_controller_factory

from cirq import circuits, ops, study
from cirq.sim import SimulatesSamples


class QasmSimulator(SimulatesSamples):
    """Contains an OpenCL based backend

    **Backend options**

    The following backend options may be used with in the
    ``backend_options`` kwarg for :meth:`QasmSimulator.run` or
    ``qiskit.execute``:

    * ``"normalize"`` (bool): Keep track of the total global probability
      normalization, and correct toward exactly 1. (Also turns on
      "zero_threshold". With "zero_threshold">0 "schmidt_decompose"=True,
      this can actually improve execution time, for opportune circuits.)

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the simulation, gate-to-gate. (Only used
      if "normalize" is enabled. Default value: Qrack default)

    * ``"schmidt_decompose"`` (bool): If true, enable "QUnit" layer of
      Qrack, including Schmidt decomposition optimizations.

    * ``"paging"`` (bool): If true, enable "QPager" layer of Qrack.

    * ``"stabilizer"`` (bool): If true, enable Qrack "QStabilizerHybrid"
      layer of Qrack. (This can be enabled with universal gate simulations.)

    * ``"opencl"`` (bool): If true, use the OpenCL engine of Qrack
      ("QEngineOCL") as the base "Schroedinger method" simulator.
      If OpenCL is not available, simulation will fall back to CPU.

    * ``"opencl_device_id"`` (int): (If OpenCL is enabled,) choose
      the OpenCl device to simulate on, (indexed by order of device
      discovery on OpenCL load/compilation). "-1" indicates to use
      the Qrack default device, (the last discovered, which tends to
      be a non-CPU accelerator, on common personal hardware systems.)
      If "opencl-multi" is active, set the default device index.

    * ``"opencl-multi"`` (bool): (If OpenCL and Schmidt decomposition
      are enabled,) distribute Schmidt-decomposed sub-engines among
      all available OpenCL devices.
    """

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': '5.4.0',
        'n_qubits': 64,
        'conditional': True,
        'url': 'https://github.com/vm6502q/qiskit-qrack-provider',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'An OpenCL based qasm simulator',
        'coupling_map': None,
        'normalize': True,
        'zero_threshold': -999.0,
        'schmidt_decompose': True,
        'paging': True,
        'stabilizer': True,
        'opencl': True,
        'opencl_device_id': -1,
        'opencl_multi': False,
        'basis_gates': [
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'cx', 'cz', 'ch', 'id', 'x', 'sx', 'y', 'z', 'h',
            'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'swap', 'ccx', 'initialize', 'cu1', 'cu2',
            'cu3', 'cswap', 'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3', 'mcswap',
            'multiplexer', 'reset', 'measure'
        ],
        'gates': [{
            'name': 'u1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate u1(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'u2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description':
            'Single-qubit gate [[1, -exp(1j*lam)], [exp(1j*phi), exp(1j*(phi+lam))]]/sqrt(2)',
            'qasm_def': 'gate u2(phi,lam) q { U(pi/2,phi,lam) q; }'
        }, {
            'name':
            'u3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional':
            True,
            'description':
            'Single-qubit gate with three rotation angles',
            'qasm_def':
            'gate u3(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name':
            'u',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional':
            True,
            'description':
            'Single-qubit gate with three rotation angles',
            'qasm_def':
            'gate u(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name': 'p',
            'parameters': ['theta', 'phi'],
            'conditional': True,
            'description': 'Single-qubit gate [[cos(theta), -1j*exp(-1j*phi)], [sin(theta), -1j*exp(1j *phi)*sin(theta), cos(theta)]]',
            'qasm_def': 'gate r(theta, phi) q { U(theta, phi - pi/2, -phi + pi/2) q;}'
        }, {
            'name': 'r',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate p(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'cx',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-NOT gate',
            'qasm_def': 'gate cx c,t { CX c,t; }'
        }, {
            'name': 'cz',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-Z gate',
            'qasm_def': 'gate cz a,b { h b; cx a,b; h b; }'
        }, {
            'name': 'ch',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-H gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'id',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit identity gate',
            'qasm_def': 'gate id a { U(0,0,0) a; }'
        }, {
            'name': 'x',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-X gate',
            'qasm_def': 'gate x a { U(pi,0,pi) a; }'
        }, {
            'name': 'sx',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit square root of Pauli-X gate',
            'qasm_def': 'gate sx a { rz(-pi/2) a; h a; rz(-pi/2); }'
        }, {
            'name': 'y',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'z',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'h',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Hadamard gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'rx',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-X axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ry',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Y axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'rz',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 's',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'sdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 't',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'tdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'swap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccx',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Toffoli gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cswap',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Fredkin (controlled-SWAP) gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'initialize',
            'parameters': ['vector'],
            'conditional': False,
            'description': 'N-qubit state initialize. '
                           'Resets qubits then sets statevector to the parameter vector.',
            'qasm_def': 'initialize(vector) q1, q2,...'
        }, {
            'name': 'cu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcx',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcy',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcz',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcswap',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'multiplexer',
            'parameters': ['mat1', 'mat2', '...'],
            'conditional': True,
            'description': 'N-qubit multi-plexer gate. '
                           'The input parameters are the gates for each value.'
                           'WARNING: Qrack currently only supports single-qubit-target multiplexer gates',
            'qasm_def': 'TODO'
        }, {
            'name': 'reset',
            'parameters': [],
            'conditional': True,
            'description': 'Reset qubit to 0 state',
            'qasm_def': 'TODO'
        }]
    }

    # TODO: Implement these __init__ options. (We only match the signature for any compatibility at all, for now.)
    def __init__(self,
                 configuration=None):
        self._configuration = configuration or self.DEFAULT_CONFIGURATION
        self._number_of_qubits = None
        self._results = {}
        self._shots = {}
        self._local_random = np.random.RandomState()

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        qubit_order = sorted(resolved_circuit.all_qubits())

        # Simulate as many unitary operations as possible before having to
        # repeat work for each sample.
        unitary_prefix, general_suffix = (
            split_into_matching_protocol_then_general(resolved_circuit, protocols.has_unitary)
            if protocols.has_unitary(self.noise)
            else (resolved_circuit[0:0], resolved_circuit)
        )
        
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(unitary_prefix.all_qubits())
        num_qubits = len(qubits)
        qid_shape = protocols.qid_shape(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}

        is_unitary_preamble = False

        if self._sample_measure:
            nonunitary_start = 0
        else:
            is_unitary_preamble = True
            self._sample_measure = True
            self._sim = qrack_controller_factory()
            self._sim.initialize_qreg(self._configuration.opencl,
                                      self._configuration.schmidt_decompose,
                                      self._configuration.paging,
                                      self._configuration.stabilizer,
                                      self._number_of_qubits,
                                      self._configuration.opencl_device_id,
                                      self._configuration.opencl_multi,
                                      self._configuration.normalize,
                                      self._configuration.zero_threshold)

            for op in unitary_prefix:
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                self._try_gate(op, indices)

            self._sample_measure = False

        preamble_sim = self._sim if is_unitary_preamble else None

        for shot in range(shotLoopMax):
            if not is_unitary_preamble:
                self._sim = qrack_controller_factory()
                self._sim.initialize_qreg(self._configuration.opencl,
                                          self._configuration.schmidt_decompose,
                                          self._configuration.paging,
                                          self._configuration.stabilizer,
                                          self._number_of_qubits,
                                          self._configuration.opencl_device_id,
                                          self._configuration.opencl_multi,
                                          self._configuration.normalize,
                                          self._configuration.zero_threshold)
            else:
                self._sim = preamble_sim.clone()

            for operation in experiment.instructions[nonunitary_start:]:
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                self._apply_op(operation, shotsPerLoop)

        return dict(Counter(self.__memory))
        
    def _try_gate(self, op: ops.GateOperation, indices: np.array):
        # One qubit gate
        if isinstance(op.gate, ops.pauli_gates._PauliX):
            self._sim.x(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliY):
            self._sim.y(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliZ):
            self._sim.z(indices[0])
        elif isinstance(op.gate, ops.common_gates.HPowGate):
            mat = np.power([[1,1],[1,-1]], -np.pi * op.gate._exponent)
            self._sim.matrix_gate(indices[0], mat)
        elif isinstance(op.gate, ops.common_gates.XPowGate):
            self._sim.rx(indices[0], [-np.pi * op.gate._exponent])
        elif isinstance(op.gate, ops.common_gates.YPowGate):
            self._sim.ry(indices[0], [-np.pi * op.gate._exponent])
        elif isinstance(op.gate, ops.common_gates.ZPowGate):
            self._sim.rz(indices[0], [-np.pi * op.gate._exponent])
        elif (len(indices) == 1 and isinstance(op.gate, ops.matrix_gates.MatrixGate)):
            mat = op.gate._matrix
            self._sim.matrix_gate(indices[0], mat)
        elif isinstance(op.gate, circuits.qasm_output.QasmUGate):
            lmda = op.gate.lmda
            theta = op.gate.theta
            phi = op.gate.phi
            self._sim.u(indices[0], [lmda, theta, phi])

        # Two qubit gate
        elif isinstance(op.gate, ops.common_gates.CNotPowGate):
            if op.gate._exponent == 1.0:
                self._sim.cx([indices[0], indices[1]])
            else:
                mat = np.power([[0,1],[1,0]], -np.pi * op.gate._exponent)
                self._sim.ctrld_matrix_gate(indices, mat)
        elif isinstance(op.gate, ops.common_gates.CZPowGate):
            if op.gate._exponent == 1.0:
                self._sim.cz([indices[0], indices[1]])
            else:
                mat = np.power([[1,0],[0,-1]], -np.pi * op.gate._exponent)
                self._sim.ctrld_matrix_gate(indices, mat)
        elif isinstance(op.gate, ops.common_gates.SwapPowGate):
            if op.gate._exponent == 1.0:
                self._sim.swap(indices[0], indices[1])
            elif op.gate._exponent == 0.5:
                self._sim.sqrtswap(indices[0], indices[1])
            else:
                return False
        #TODO:
        #elif isinstance(op.gate, ops.parity_gates.XXPowGate):
        #    qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [1, 1], -np.pi * op.gate._exponent)
        #elif isinstance(op.gate, ops.parity_gates.YYPowGate):
        #    qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [2, 2], -np.pi * op.gate._exponent)
        #elif isinstance(op.gate, ops.parity_gates.ZZPowGate):
        #    qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [3, 3], -np.pi * op.gate._exponent)
        #elif (len(indices) == 2 and isinstance(op.gate, ops.matrix_gates.MatrixGate)):
        #    indices.reverse()
        #    mat = op.gate._matrix
        #    qulacs_circuit.add_dense_matrix_gate(indices, mat)

        # Three qubit gate
        elif isinstance(op.gate, ops.three_qubit_gates.CCXPowGate):
            if op.gate._exponent == 1.0:
                self._sim.ccx([indices[0], indices[1]])
            else:
                mat = np.power([[0,1],[1,0]], -np.pi * op.gate._exponent)
                self._sim.ctrld_matrix_gate(indices, mat)
        elif isinstance(op.gate, ops.three_qubit_gates.CCZPowGate):
            if op.gate._exponent == 1.0:
                self._sim.ccz([indices[0], indices[1]])
            else:
                mat = np.power([[0,1],[1,0]], -np.pi * op.gate._exponent)
                self._sim.ctrld_matrix_gate(indices, mat)
        elif isinstance(op.gate, ops.three_qubit_gates.CSwapGate):
            self._sim.cswap(indices)

        # Misc
        #elif protocols.has_unitary(op):
        #    indices.reverse()
        #    mat = op._unitary_()
        #    qulacs_circuit.add_dense_matrix_gate(indices, mat)

        # Not unitary
        else:
            return False

        return True
