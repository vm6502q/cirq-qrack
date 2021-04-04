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
        'opencl_multi': False
    }

    # TODO: Implement these __init__ options. (We only match the signature for any compatibility at all, for now.)
    def __init__(self,
                 configuration=None):
        self._configuration = configuration or self.DEFAULT_CONFIGURATION
        self._number_of_qubits = None
        self._results = {}
        self._shots = {}
        self._classical_memory = 0
        self._local_random = np.random.RandomState()

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
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

        is_unitary_preamble = False

        self._sample_measure = True
        self._sim = qrack_controller_factory()
        self._sim.initialize_qreg(self._configuration['opencl'],
                                  self._configuration['schmidt_decompose'],
                                  self._configuration['paging'],
                                  self._configuration['stabilizer'],
                                  self._number_of_qubits,
                                  self._configuration['opencl_device_id'],
                                  self._configuration['opencl_multi'],
                                  self._configuration['normalize'],
                                  self._configuration['zero_threshold'])

        for moment in unitary_prefix:
            operations = moment.operations
            for op in operations:
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                self._try_gate(op, indices)

        general_ops = list(general_suffix.all_operations())
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
            self.__memory = self._add_sample_measure(indices, repetitions)
            return dict(Counter(self.__memory))

        self._sample_measure = False

        for shot in range(repetitions):
            loopSuffix = general_suffix
            self._sim = preamble_sim.clone()

            for moment in general_suffix:
                operations = moment.operations
                for op in operations:
                    indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                    self._try_gate(op, indices)

        return dict(Counter(self.__memory))
        
    def _try_gate(self, op: ops.GateOperation, indices: np.array):
        # One qubit gate
        if isinstance(op.gate, ops.pauli_gates._PauliX):
            self._sim.x([indices[0]])
        elif isinstance(op.gate, ops.pauli_gates._PauliY):
            self._sim.y([indices[0]])
        elif isinstance(op.gate, ops.pauli_gates._PauliZ):
            self._sim.z([indices[0]])
        elif isinstance(op.gate, ops.common_gates.HPowGate):
            mat = np.power([[1,1],[1,-1]], -np.pi * op.gate._exponent)
            self._sim.matrix_gate([indices[0]], mat)
        elif isinstance(op.gate, ops.common_gates.XPowGate):
            self._sim.rx([indices[0]], [-np.pi * op.gate._exponent])
        elif isinstance(op.gate, ops.common_gates.YPowGate):
            self._sim.ry([indices[0]], [-np.pi * op.gate._exponent])
        elif isinstance(op.gate, ops.common_gates.ZPowGate):
            self._sim.rz([indices[0]], [-np.pi * op.gate._exponent])
        elif (len(indices) == 1 and isinstance(op.gate, ops.matrix_gates.MatrixGate)):
            mat = op.gate._matrix
            self._sim.matrix_gate([indices[0]], mat)
        elif isinstance(op.gate, circuits.qasm_output.QasmUGate):
            lmda = op.gate.lmda
            theta = op.gate.theta
            phi = op.gate.phi
            self._sim.u([indices[0]], [lmda, theta, phi])

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
                self._sim.cx([indices[0], indices[1], indices[2]])
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

    def _add_sample_measure(self, measure_qubit, num_samples):
        """Generate memory samples from current statevector.
        Taken almost straight from the terra source code.
        Args:
            measure_params (list): List of (qubit, clbit) values for
                                   measure instructions to sample.
            num_samples (int): The number of memory samples to generate.
        Returns:
            list: A list of memory values in hex format.
        """
        memory = []

        # If we only want one sample, it's faster for the backend to do it,
        # without passing back the probabilities.
        if num_samples == 1:
            sample = self._sim.measure(measure_qubit)
            classical_state = self._classical_memory
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                qubit_outcome = (sample >> qubit) & 1
                classical_state = (classical_state & (~1)) | qubit_outcome
            outKey = bin(classical_state)[2:]
            memory += [hex(int(outKey, 2))]
            self._classical_memory = classical_state
            return memory

        # Sample and convert to bit-strings
        measure_results = self._sim.measure_shots(measure_qubit, num_samples)
        classical_state = self._classical_memory
        for key, value in measure_results.items():
            sample = key
            classical_state = self._classical_memory
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                qubit_outcome = (sample >> index) & 1
                classical_state = (classical_state & (~1)) | qubit_outcome
            outKey = bin(classical_state)[2:]
            memory += value * [hex(int(outKey, 2))]

        return memory
