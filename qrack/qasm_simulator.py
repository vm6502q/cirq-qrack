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
import scipy as sp
import collections
from typing import Dict
from pyqrack import QrackSimulator, Pauli

import cirq
from cirq import circuits, ops, protocols, study
from cirq.sim import SimulatesSamples
from cirq.sim.simulator import check_all_resolved, split_into_matching_protocol_then_general


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
        'url': 'https://github.com/vm6502q/cirq-qrack',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'An Schmidt-decomposed, OpenCL-based QASM simulator',
        'coupling_map': None,
        'schmidt_decompose': True,
        'paging': True,
        'stabilizer': True,
        'qbdt': False,
        'opencl': True,
        'opencl_multi': True,
        'mask_fusion_1qb': False,
        'hybrid_opencl': True,
        'host_pointer': False
    }

    # TODO: Implement these __init__ options. (We only match the signature for any compatibility at all, for now.)
    def __init__(self,
                 configuration=None):
        self._configuration = configuration or self.DEFAULT_CONFIGURATION
        self._number_of_qubits = None
        self._memory = collections.defaultdict(list)
        self._results = {}
        self._shots = {}
        self._local_random = np.random.RandomState()

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        qubit_order = sorted(resolved_circuit.all_qubits())

        self._number_of_qubits = len(qubit_order)

        # Simulate as many unitary operations as possible before having to
        # repeat work for each sample.
        unitary_prefix, general_suffix = (
            split_into_matching_protocol_then_general(resolved_circuit, protocols.has_unitary)
        )
        
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(unitary_prefix.all_qubits())
        num_qubits = len(qubits)
        qid_shape = protocols.qid_shape(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}

        self._sample_measure = True
        self._sim = QrackSimulator(self._number_of_qubits,
                                   isSchmidtDecomposeMulti=self._configuration['opencl_multi'],
                                   isSchmidtDecompose=self._configuration['schmidt_decompose'],
                                   isStabilizerHybrid=self._configuration['stabilizer'],
                                   isBinaryDecisionTree=self._configuration['qbdt'],
                                   isPaged=self._configuration['paging'],
                                   is1QbFusion=self._configuration['mask_fusion_1qb'],
                                   isCpuGpuHybrid=self._configuration['hybrid_opencl'],
                                   isOpenCL=self._configuration['opencl'],
                                   isHostPointer=self._configuration['host_pointer'])

        for moment in unitary_prefix:
            operations = moment.operations
            for op in operations:
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                if not self._try_gate(op, indices):
                    raise RuntimeError("Qrack can't perform your gate: " + str(op))

        general_ops = list(general_suffix.all_operations())
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            indices = []
            for op in general_ops:
                indices = indices + [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
            sample_measure = self._add_sample_measure(indices, repetitions)
            for sample in sample_measure:
                qb_index = 0
                for op in general_ops:
                    key = protocols.measurement_key_name(op.gate)
                    value = []
                    for _ in op.qubits:
                        value.append(sample[qb_index])
                        qb_index = qb_index + 1
                    self._memory[key].append(np.asarray([value]))

            __memory = {}
            for key, value in self._memory.items():
                __memory[key] = np.asarray(value)
            self._memory = __memory

            return self._memory

        self._sample_measure = False
        preamble_sim = self._sim

        for shot in range(repetitions):
            self._sim = preamble_sim.clone()
            for moment in general_suffix:
                operations = moment.operations
                for op in operations:
                    indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                    key = protocols.measurement_key_name(op.gate)
                    self._memory[key].append(np.asarray([self._add_qasm_measure(indices)]))

        __memory = {}
        for key, value in self._memory.items():
            __memory[key] = np.asarray(value)
        self._memory = __memory

        return self._memory
        
    def _try_gate(self, op: ops.GateOperation, indices: np.array):
        # One qubit gate
        if isinstance(op.gate, ops.pauli_gates._PauliX):
            self._sim.x(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliY):
            self._sim.y(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliZ):
            self._sim.z(indices[0])
        elif isinstance(op.gate, ops.common_gates.HPowGate):
            if op.gate._exponent == 1.0:
                self._sim.h(indices[0])
            else :
                c = np.cos(np.pi * t / 2.0)
                s = np.sin(np.pi * t / 2.0)
                g = np.exp((np.pi * t / 2.0) * (1.0j))
                mat = [g * (c - (1.0j) * s / sqrt(2.0)), -(1.0j) * g * s / sqrt(2.0), -(1.0j) * g * s / sqrt(2.0), g * (c + (1.0j) * s / sqrt(2.0))]
                self._sim.mtrx(mat, indices[0])
        elif isinstance(op.gate, ops.common_gates.XPowGate):
            self._sim.r(Pauli.PauliX, -np.pi * op.gate._exponent, indices[0])
        elif isinstance(op.gate, ops.common_gates.YPowGate):
            self._sim.r(Pauli.PauliY, -np.pi * op.gate._exponent, indices[0])
        elif isinstance(op.gate, ops.common_gates.ZPowGate):
            self._sim.r(Pauli.PauliZ, -np.pi * op.gate._exponent, indices[0])
        elif isinstance(op.gate, ops.PhasedXPowGate):
            mat = cirq.unitary(op.gate)
            self._sim.mtrx([item for sublist in mat for item in sublist], indices[0])
        elif (len(indices) == 1 and isinstance(op.gate, ops.matrix_gates.MatrixGate)):
            mat = op.gate._matrix
            self._sim.mtrx(mat, indices[0])
        elif isinstance(op.gate, circuits.qasm_output.QasmUGate):
            lmda = op.gate.lmda
            theta = op.gate.theta
            phi = op.gate.phi
            self._sim.u(indices[0], theta * np.pi, phi * np.pi, lmda * np.pi)

        # Two qubit gate
        elif isinstance(op.gate, ops.common_gates.CNotPowGate):
            if op.gate._exponent == 1.0:
                self._sim.mcx(indices[0:1], indices[1])
            else:
                mat = sp.linalg.fractional_matrix_power([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], -np.pi * op.gate._exponent)
                self._sim.mcmtrx(indices[0:1], [item for sublist in mat for item in sublist], indices[1])
        elif isinstance(op.gate, ops.common_gates.CZPowGate):
            if op.gate._exponent == 1.0:
                self._sim.mcz(indices[0:1], indices[1])
            else:
                mat = sp.linalg.fractional_matrix_power([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], -np.pi * op.gate._exponent)
                self._sim.mcmtrx(indices[0:1], [item for sublist in mat for item in sublist], indices[1])
        elif isinstance(op.gate, ops.common_gates.SwapPowGate):
            if op.gate._exponent == 1.0:
                self._sim.swap(indices[0], indices[1])
            elif op.gate._exponent == 0.5:
                self._sim.sqrtswap(indices[0], indices[1])
            else:
                return False
        elif isinstance(op.gate, ops.FSimGate):
            theta = op.gate.theta
            phi = op.gate.phi
            self._sim.fsim(theta, phi, indices[0], indices[1])
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
                self._sim.mcx(indices[0:2], indices[2])
            else:
                mat = sp.linalg.fractional_matrix_power([[0.0 + 0.0j, 1.0 + 0.0j],[1.0 + 0.0j, 0.0 + 0.0j]], -np.pi * op.gate._exponent)
                self._sim.mcmtrx(indices[0:2], [item for sublist in mat for item in sublist], indices[2])
        elif isinstance(op.gate, ops.three_qubit_gates.CCZPowGate):
            if op.gate._exponent == 1.0:
                self._sim.mcz(indices[0:2], indices[2])
            else:
                mat = sp.linalg.fractional_matrix_power([[0.0 + 0.0j, 1.0 + 0.0j],[1.0 + 0.0j, 0.0 + 0.0j]], -np.pi * op.gate._exponent)
                self._sim.mcmtrx(indices[0:2], [item for sublist in mat for item in sublist], indices[2])
        elif isinstance(op.gate, ops.three_qubit_gates.CSwapGate):
            self._sim.cswap(indices[0:1], indices[1], indices[2])

        # Misc
        #elif protocols.has_unitary(op):
        #    indices.reverse()
        #    mat = op._unitary_()
        #    qulacs_circuit.add_dense_matrix_gate(indices, mat)

        # Not unitary
        else:
            return False

        return True

    def _add_sample_measure(self, measure_qubit, num_samples):
        """Generate memory samples from current statevector.
        Taken almost straight from the terra source code.
        Args:
            measure_qubit (int[]): qubits to be measured.
            num_samples (int): The number of memory samples to generate.
        Returns:
            list: A list of memory values.
        """

        # If we only want one sample, it's faster for the backend to do it,
        # without passing back the probabilities.
        if num_samples == 1:
            key = 0
            for i in range(len(measure_qubit)):
                if self._sim.m(measure_qubit[i]):
                    key = key | (1 << i)
            return np.asarray([self._int_to_bits(key, len(measure_qubit))])

        # Sample and convert to bit-strings
        memory = []
        measure_results = self._sim.measure_shots(measure_qubit, num_samples)
        for value in measure_results:
            memory += [self._int_to_bits(int(value), len(measure_qubit))]

        return np.asarray(memory)

    def _add_qasm_measure(self, measure_qubit):
        """Apply a measure instruction to a qubit.
        Args:
            measure_qubit (int[]): qubits to be measured.
        Returns:
            int: Memory values.
        """

        key = self._sim.m(measure_qubit)
        return self._int_to_bits(int(key), len(measure_qubit))

    def _int_to_bits(self, i, len):
        bits = []
        for _ in range(len):
            bits.append(i & 1)
            i = i >> 1
        return bits
