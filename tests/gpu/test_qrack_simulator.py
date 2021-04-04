import re
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.stats import unitary_group

import cirq
from qrack import QasmSimulator


def parse_qasm_to_QrackCircuit(input_filename, cirq_circuit, cirq_qubits):
    with open(input_filename, "r") as ifile:
        lines = ifile.readlines()

        for line in lines:
            s = re.search(r"qreg|cx|u3|u1", line)

            if s is None:
                continue

            elif s.group() == 'qreg':
                match = re.search(r"\d\d*", line)
                # print(match)
                continue

            elif s.group() == 'cx':
                match = re.findall(r'\[\d\d*\]', line)
                c_qbit = int(match[0].strip('[]'))
                t_qbit = int(match[1].strip('[]'))
                cirq_circuit.append(cirq.ops.CNOT(cirq_qubits[c_qbit], cirq_qubits[t_qbit]))
                continue

            elif s.group() == 'u3':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)
                # cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]),float(m_r[1]),float(m_r[2])).on(cirq_qubits[int(m_i[0].strip('[]'))]))
                target_index = int(m_i[0].strip('[]'))
                cirq_circuit.append(cirq.rz(float(m_r[0])).on(cirq_qubits[target_index]))
                cirq_circuit.append(cirq.ry(float(m_r[1])).on(cirq_qubits[target_index]))
                cirq_circuit.append(cirq.rz(float(m_r[2])).on(cirq_qubits[target_index]))
                continue

            elif s.group() == 'u1':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)

                # cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]), 0, 0).on(cirq_qubits[int(m_i[0].strip('[]'))]))
                target_index = int(m_i[0].strip('[]'))
                cirq_circuit.append(cirq.rz(float(m_r[0])).on(cirq_qubits[target_index]))
                continue


class TestQrackSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qubit_n = 5
        self.test_repeat = 4

    def check_result(self, circuit, rtol=1e-9, atol=0):
        qrack_result = QasmSimulator().run(circuit, repetitions=100)
        cirq_result = cirq.Simulator(dtype=dtype).run(circuit, repetitions=100)
        self.assertEquals(actual, expected)

    def check_single_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            circuit.append(gate_op(qubits[index]))
            # print("flip {}".format(index))
            self.check_result(circuit)

    def check_single_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand() * np.pi * 2
            circuit.append(gate_op(angle).on(qubits[index]))
            self.check_result(circuit)

    def check_two_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3) * np.pi * 2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            circuit.append(gate_op(qubits[index[0]], qubits[index[1]]))
            self.check_result(circuit)

    def check_two_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3) * np.pi * 2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            angle = np.random.rand() * np.pi * 2
            gate_op_angle = gate_op(exponent=angle)
            circuit.append(gate_op_angle(qubits[index[0]], qubits[index[1]]))
            self.check_result(circuit)

    def check_three_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3) * np.pi * 2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:3]
            circuit.append(gate_op(qubits[index[0]], qubits[index[1]], qubits[index[2]]))
            self.check_result(circuit)

    def check_three_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3) * np.pi * 2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:3]
            angle = np.random.rand() * np.pi * 2
            gate_op_angle = gate_op(exponent=angle)
            circuit.append(gate_op_angle(qubits[index[0]], qubits[index[1]], qubits[index[2]]))
            self.check_result(circuit)

    def test_QrackSimulator_Xgate(self):
        self.check_single_qubit_gate(cirq.ops.X)

    def test_QrackSimulator_Ygate(self):
        self.check_single_qubit_gate(cirq.ops.Y)

    def test_QrackSimulator_Zgate(self):
        self.check_single_qubit_gate(cirq.ops.Z)

    def test_QrackSimulator_Hgate(self):
        self.check_single_qubit_gate(cirq.ops.H)

    def test_QrackSimulator_Sgate(self):
        self.check_single_qubit_gate(cirq.ops.S)

    def test_QrackSimulator_Tgate(self):
        self.check_single_qubit_gate(cirq.ops.T)

    def test_QrackSimulator_RXgate(self):
        self.check_single_qubit_rotation_gate(cirq.rx)

    def test_QrackSimulator_RYgate(self):
        self.check_single_qubit_rotation_gate(cirq.ry)

    def test_QrackSimulator_RZgate(self):
        self.check_single_qubit_rotation_gate(cirq.rz)

    def test_QrackSimulator_CNOTgate(self):
        self.check_two_qubit_gate(cirq.ops.CNOT)

    def test_QrackSimulator_CZgate(self):
        self.check_two_qubit_gate(cirq.ops.CZ)

    def test_QrackSimulator_SWAPgate(self):
        self.check_two_qubit_gate(cirq.ops.SWAP)

    def test_QrackSimulator_ISWAPgate(self):
        self.check_two_qubit_gate(cirq.ops.ISWAP)

    #def test_QrackSimulator_XXgate(self):
    #    self.check_two_qubit_gate(cirq.ops.XX)

    #def test_QrackSimulator_YYgate(self):
    #    self.check_two_qubit_gate(cirq.ops.YY)

    #def test_QrackSimulator_ZZgate(self):
    #    self.check_two_qubit_gate(cirq.ops.ZZ)

    def test_QrackSimulator_CNotPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.CNotPowGate)

    def test_QrackSimulator_CZPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.CZPowGate)

    def test_QrackSimulator_SwapPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.SwapPowGate)

    def test_QrackSimulator_ISwapPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.ISwapPowGate)

    #def test_QrackSimulator_XXPowgate(self):
    #    self.check_two_qubit_rotation_gate(cirq.ops.XXPowGate)

    #def test_QrackSimulator_YYPowgate(self):
    #    self.check_two_qubit_rotation_gate(cirq.ops.YYPowGate)

    #def test_QrackSimulator_ZZPowgate(self):
    #    self.check_two_qubit_rotation_gate(cirq.ops.ZZPowGate)

    def test_QrackSimulator_CCXgate(self):
        self.check_three_qubit_gate(cirq.ops.CCX)

    def test_QrackSimulator_CCZgate(self):
        self.check_three_qubit_gate(cirq.ops.CCZ)

    def test_QrackSimulator_CSwapGate(self):
        self.check_three_qubit_gate(cirq.ops.CSWAP)

    def test_QrackSimulator_TOFFOLIgate(self):
        self.check_three_qubit_gate(cirq.ops.TOFFOLI)

    def test_QrackSimulator_CCXPowgate(self):
        self.check_three_qubit_rotation_gate(cirq.ops.CCXPowGate)

    def test_QrackSimulator_CCZPowgate(self):
        self.check_three_qubit_rotation_gate(cirq.ops.CCZPowGate)

    def test_QrackSimulator_Ugate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand(3) * np.pi * 2
            circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            self.check_result(circuit)

    def test_QrackSimulator_SingleQubitMatrixGate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3) * np.pi * 2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            index = np.random.randint(self.qubit_n)
            mat = unitary_group.rvs(2)
            circuit.append(cirq.MatrixGate(mat).on(qubits[index]))
            self.check_result(circuit)

    def test_QrackSimulator_TwoQubitMatrixGate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            mat = unitary_group.rvs(4)
            circuit.append(cirq.MatrixGate(mat).on(qubits[index[0]], qubits[index[1]]))
            self.check_result(circuit)

    def test_QrackSimulator_QuantumVolume(self):
        qubit_n = 20
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        parse_qasm_to_QrackCircuit('tests/quantum_volume_n10_d8_0_0.qasm', circuit, qubits)
        self.check_result(circuit)


if __name__ == "__main__":
    unittest.main()
