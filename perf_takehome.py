"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            return [{engine: [slot]} for engine, slot in slots]

        instrs = []
        current = {}
        counts = defaultdict(int)

        for engine, slot in slots:
            limit = SLOT_LIMITS[engine]
            if counts[engine] >= limit:
                instrs.append(current)
                current = {}
                counts = defaultdict(int)
            current.setdefault(engine, []).append(slot)
            counts[engine] += 1

        if current:
            instrs.append(current)
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2):
        instrs = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            instrs.append(
                {
                    "alu": [
                        (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                        (op3, tmp2, val_hash_addr, self.scratch_const(val3)),
                    ]
                }
            )
            instrs.append({"alu": [(op2, val_hash_addr, tmp1, tmp2)]})

        return instrs

    def build_hash_vec(self, val_vec, tmp1_vec, tmp2_vec, const_vecs):
        instrs = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            val1_vec = const_vecs[val1]
            val3_vec = const_vecs[val3]
            instrs.append(
                {
                    "valu": [
                        (op1, tmp1_vec, val_vec, val1_vec),
                        (op3, tmp2_vec, val_vec, val3_vec),
                    ]
                }
            )
            instrs.append({"valu": [(op2, val_vec, tmp1_vec, tmp2_vec)]})

        return instrs

    def build_hash_vec_pair(
        self,
        val_vec_a,
        tmp1_vec_a,
        tmp2_vec_a,
        val_vec_b,
        tmp1_vec_b,
        tmp2_vec_b,
        const_vecs,
    ):
        instrs = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            val1_vec = const_vecs[val1]
            val3_vec = const_vecs[val3]
            instrs.append(
                {
                    "valu": [
                        (op1, tmp1_vec_a, val_vec_a, val1_vec),
                        (op3, tmp2_vec_a, val_vec_a, val3_vec),
                        (op1, tmp1_vec_b, val_vec_b, val1_vec),
                        (op3, tmp2_vec_b, val_vec_b, val3_vec),
                    ]
                }
            )
            instrs.append(
                {
                    "valu": [
                        (op2, val_vec_a, tmp1_vec_a, tmp2_vec_a),
                        (op2, val_vec_b, tmp1_vec_b, tmp2_vec_b),
                    ]
                }
            )

        return instrs

    def broadcast_const(self, scalar_addr, name=None):
        vec_addr = self.alloc_scratch(name, VLEN)
        self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
        return vec_addr

    def emit(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit_group(self, engine, slots):
        self.instrs.append({engine: slots})

    def emit_bundle(self, bundles):
        self.instrs.append(bundles)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        debug_round_writes: bool = False,
        fast_mode: bool = True,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vec_zero = self.broadcast_const(zero_const, "vec_zero")
        vec_one = self.broadcast_const(one_const, "vec_one")
        vec_two = self.broadcast_const(two_const, "vec_two")
        vec_n_nodes = self.broadcast_const(self.scratch["n_nodes"], "vec_n_nodes")
        vec_forest_values = self.broadcast_const(
            self.scratch["forest_values_p"], "vec_forest_values_p"
        )

        hash_consts = {}
        for _, val1, _, _, val3 in HASH_STAGES:
            for val in (val1, val3):
                if val not in hash_consts:
                    hash_consts[val] = self.broadcast_const(
                        self.scratch_const(val), f"vec_hash_{val}"
                    )

        if not fast_mode:
            parity_mask = self.scratch_const(1, "parity_mask")
            cache_invalid = self.scratch_const(0xFFFFFFFF, "cache_invalid")
            # Pause instructions are matched up with yield statements in the reference
            # kernel to let you debug at intermediate steps. The testing harness in this
            # file requires these match up to the reference kernel's yields, but the
            # submission harness ignores them.
            self.add("flow", ("pause",))
            # Any debug engine instruction is ignored by the submission simulator
            self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_miss_val = self.alloc_scratch("tmp_miss_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        debug_idx_ptr = self.alloc_scratch("debug_idx_ptr")
        debug_val_ptr = self.alloc_scratch("debug_val_ptr")

        cache_tag0 = self.alloc_scratch("cache_tag0")
        cache_val0 = self.alloc_scratch("cache_val0")
        cache_tag1 = self.alloc_scratch("cache_tag1")
        cache_val1 = self.alloc_scratch("cache_val1")
        cache_toggle = self.alloc_scratch("cache_toggle")

        vec_node_val = self.alloc_scratch("vec_node_val", VLEN)
        vec_addr = self.alloc_scratch("vec_addr", VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_node_val2 = self.alloc_scratch("vec_node_val2", VLEN)
        vec_addr2 = self.alloc_scratch("vec_addr2", VLEN)
        vec_tmp1b = self.alloc_scratch("vec_tmp1b", VLEN)
        vec_tmp2b = self.alloc_scratch("vec_tmp2b", VLEN)
        vec_node_val3 = self.alloc_scratch("vec_node_val3", VLEN)
        vec_addr3 = self.alloc_scratch("vec_addr3", VLEN)
        vec_tmp1c = self.alloc_scratch("vec_tmp1c", VLEN)
        vec_tmp2c = self.alloc_scratch("vec_tmp2c", VLEN)
        vec_node_val4 = self.alloc_scratch("vec_node_val4", VLEN)
        vec_addr4 = self.alloc_scratch("vec_addr4", VLEN)
        vec_tmp1d = self.alloc_scratch("vec_tmp1d", VLEN)
        vec_tmp2d = self.alloc_scratch("vec_tmp2d", VLEN)
        vec_node_val5 = self.alloc_scratch("vec_node_val5", VLEN)
        vec_addr5 = self.alloc_scratch("vec_addr5", VLEN)
        vec_tmp1e = self.alloc_scratch("vec_tmp1e", VLEN)
        vec_tmp2e = self.alloc_scratch("vec_tmp2e", VLEN)
        vec_node_val6 = self.alloc_scratch("vec_node_val6", VLEN)
        vec_addr6 = self.alloc_scratch("vec_addr6", VLEN)
        vec_tmp1f = self.alloc_scratch("vec_tmp1f", VLEN)
        vec_tmp2f = self.alloc_scratch("vec_tmp2f", VLEN)
        vec_node_val7 = self.alloc_scratch("vec_node_val7", VLEN)
        vec_addr7 = self.alloc_scratch("vec_addr7", VLEN)
        vec_tmp1g = self.alloc_scratch("vec_tmp1g", VLEN)
        vec_tmp2g = self.alloc_scratch("vec_tmp2g", VLEN)
        vec_node_val8 = self.alloc_scratch("vec_node_val8", VLEN)
        vec_addr8 = self.alloc_scratch("vec_addr8", VLEN)
        vec_tmp1h = self.alloc_scratch("vec_tmp1h", VLEN)
        vec_tmp2h = self.alloc_scratch("vec_tmp2h", VLEN)

        idx_buf = self.alloc_scratch("idx_buf", batch_size)
        val_buf = self.alloc_scratch("val_buf", batch_size)

        if not fast_mode:
            self.emit("alu", ("+", cache_tag0, cache_invalid, zero_const))
            self.emit("alu", ("+", cache_val0, zero_const, zero_const))
            self.emit("alu", ("+", cache_tag1, cache_invalid, zero_const))
            self.emit("alu", ("+", cache_val1, zero_const, zero_const))
            self.emit("alu", ("+", cache_toggle, zero_const, zero_const))

        vec_end = batch_size - (batch_size % VLEN)
        vec_unroll = 8 * VLEN
        vec_unroll_end = vec_end - (vec_end % vec_unroll)
        tail_len = batch_size - vec_end

        idx_load_ptr = self.alloc_scratch("idx_load_ptr")
        val_load_ptr = self.alloc_scratch("val_load_ptr")
        idx_store_ptr = self.alloc_scratch("idx_store_ptr")
        val_store_ptr = self.alloc_scratch("val_store_ptr")

        def emit_hash_with_prefetch(
            hash_instrs,
            prefetch_addr_a=None,
            prefetch_addr_b=None,
            prefetch_vec_a=None,
            prefetch_vec_b=None,
        ):
            instrs = []
            for idx, instr in enumerate(hash_instrs):
                new_instr = {key: slots.copy() for key, slots in instr.items()}
                if prefetch_addr_a is not None and idx < VLEN:
                    load_slots = new_instr.setdefault("load", [])
                    load_slots.append(
                        ("load_offset", prefetch_vec_a, prefetch_addr_a, idx)
                    )
                    load_slots.append(
                        ("load_offset", prefetch_vec_b, prefetch_addr_b, idx)
                    )
                instrs.append(new_instr)
            self.instrs.extend(instrs)

        def emit_vec_block(i):
            idx_addr = idx_buf + i
            val_addr = val_buf + i
            self.emit("valu", ("+", vec_addr, idx_addr, vec_forest_values))
            for lane in range(0, VLEN, 2):
                self.emit_group(
                    "load",
                    [
                        ("load_offset", vec_node_val, vec_addr, lane),
                        ("load_offset", vec_node_val, vec_addr, lane + 1),
                    ],
                )
            self.emit("valu", ("^", val_addr, val_addr, vec_node_val))
            self.instrs.extend(
                self.build_hash_vec(val_addr, vec_tmp1, vec_tmp2, hash_consts)
            )
            self.emit("valu", ("&", vec_tmp1, val_addr, vec_one))
            self.emit("valu", ("+", vec_tmp2, vec_tmp1, vec_one))
            self.emit("valu", ("*", idx_addr, idx_addr, vec_two))
            self.emit("valu", ("+", idx_addr, idx_addr, vec_tmp2))
            self.emit("valu", ("<", vec_tmp1, idx_addr, vec_n_nodes))
            self.emit("valu", ("*", idx_addr, idx_addr, vec_tmp1))

        self.emit("alu", ("+", idx_load_ptr, self.scratch["inp_indices_p"], zero_const))
        self.emit("alu", ("+", val_load_ptr, self.scratch["inp_values_p"], zero_const))
        for i in range(0, vec_end, VLEN):
            idx_addr = idx_buf + i
            val_addr = val_buf + i
            self.emit("load", ("vload", idx_addr, idx_load_ptr))
            self.emit("load", ("vload", val_addr, val_load_ptr))
            self.emit("flow", ("add_imm", idx_load_ptr, idx_load_ptr, VLEN))
            self.emit("flow", ("add_imm", val_load_ptr, val_load_ptr, VLEN))

        if fast_mode and tail_len:
            idx_addr = idx_buf + vec_end
            val_addr = val_buf + vec_end
            for lane in range(tail_len):
                self.emit("load", ("load", idx_addr + lane, idx_load_ptr))
                self.emit("load", ("load", val_addr + lane, val_load_ptr))
                self.emit("flow", ("add_imm", idx_load_ptr, idx_load_ptr, 1))
                self.emit("flow", ("add_imm", val_load_ptr, val_load_ptr, 1))
            for lane in range(tail_len, VLEN):
                self.emit("alu", ("+", idx_addr + lane, zero_const, zero_const))
                self.emit("alu", ("+", val_addr + lane, zero_const, zero_const))
        elif not fast_mode:
            for i in range(vec_end, batch_size):
                idx_addr = idx_buf + i
                val_addr = val_buf + i
                self.emit("load", ("load", idx_addr, idx_load_ptr))
                self.emit("load", ("load", val_addr, val_load_ptr))
                self.emit("flow", ("add_imm", idx_load_ptr, idx_load_ptr, 1))
                self.emit("flow", ("add_imm", val_load_ptr, val_load_ptr, 1))

        for round in range(rounds):
            if debug_round_writes:
                self.emit(
                    "alu",
                    ("+", debug_idx_ptr, self.scratch["inp_indices_p"], zero_const),
                )
                self.emit(
                    "alu",
                    ("+", debug_val_ptr, self.scratch["inp_values_p"], zero_const),
                )
            for i in range(0, vec_unroll_end, vec_unroll):
                idx_addr = idx_buf + i
                idx_addr2 = idx_buf + i + VLEN
                idx_addr3 = idx_buf + i + 2 * VLEN
                idx_addr4 = idx_buf + i + 3 * VLEN
                idx_addr5 = idx_buf + i + 4 * VLEN
                idx_addr6 = idx_buf + i + 5 * VLEN
                idx_addr7 = idx_buf + i + 6 * VLEN
                idx_addr8 = idx_buf + i + 7 * VLEN
                val_addr = val_buf + i
                val_addr2 = val_buf + i + VLEN
                val_addr3 = val_buf + i + 2 * VLEN
                val_addr4 = val_buf + i + 3 * VLEN
                val_addr5 = val_buf + i + 4 * VLEN
                val_addr6 = val_buf + i + 5 * VLEN
                val_addr7 = val_buf + i + 6 * VLEN
                val_addr8 = val_buf + i + 7 * VLEN

                self.emit_bundle(
                    {
                        "valu": [
                            ("+", vec_addr, idx_addr, vec_forest_values),
                            ("+", vec_addr2, idx_addr2, vec_forest_values),
                            ("+", vec_addr3, idx_addr3, vec_forest_values),
                            ("+", vec_addr4, idx_addr4, vec_forest_values),
                            ("+", vec_addr5, idx_addr5, vec_forest_values),
                            ("+", vec_addr6, idx_addr6, vec_forest_values),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("+", vec_addr7, idx_addr7, vec_forest_values),
                            ("+", vec_addr8, idx_addr8, vec_forest_values),
                        ]
                    }
                )
                for lane in range(VLEN):
                    self.emit_bundle(
                        {
                            "load": [
                                ("load_offset", vec_node_val, vec_addr, lane),
                                ("load_offset", vec_node_val2, vec_addr2, lane),
                            ]
                        }
                    )
                self.emit_bundle(
                    {
                        "valu": [
                            ("^", val_addr, val_addr, vec_node_val),
                            ("^", val_addr2, val_addr2, vec_node_val2),
                        ]
                    }
                )
                hash01 = self.build_hash_vec_pair(
                    val_addr,
                    vec_tmp1,
                    vec_tmp2,
                    val_addr2,
                    vec_tmp1b,
                    vec_tmp2b,
                    hash_consts,
                )
                emit_hash_with_prefetch(
                    hash01, vec_addr3, vec_addr4, vec_node_val3, vec_node_val4
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("^", val_addr3, val_addr3, vec_node_val3),
                            ("^", val_addr4, val_addr4, vec_node_val4),
                        ]
                    }
                )
                hash23 = self.build_hash_vec_pair(
                    val_addr3,
                    vec_tmp1c,
                    vec_tmp2c,
                    val_addr4,
                    vec_tmp1d,
                    vec_tmp2d,
                    hash_consts,
                )
                emit_hash_with_prefetch(
                    hash23, vec_addr5, vec_addr6, vec_node_val5, vec_node_val6
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("^", val_addr5, val_addr5, vec_node_val5),
                            ("^", val_addr6, val_addr6, vec_node_val6),
                        ]
                    }
                )
                hash45 = self.build_hash_vec_pair(
                    val_addr5,
                    vec_tmp1e,
                    vec_tmp2e,
                    val_addr6,
                    vec_tmp1f,
                    vec_tmp2f,
                    hash_consts,
                )
                emit_hash_with_prefetch(
                    hash45, vec_addr7, vec_addr8, vec_node_val7, vec_node_val8
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("^", val_addr7, val_addr7, vec_node_val7),
                            ("^", val_addr8, val_addr8, vec_node_val8),
                        ]
                    }
                )
                self.instrs.extend(
                    self.build_hash_vec_pair(
                        val_addr7,
                        vec_tmp1g,
                        vec_tmp2g,
                        val_addr8,
                        vec_tmp1h,
                        vec_tmp2h,
                        hash_consts,
                    )
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("&", vec_tmp1, val_addr, vec_one),
                            ("&", vec_tmp1b, val_addr2, vec_one),
                            ("&", vec_tmp1c, val_addr3, vec_one),
                            ("&", vec_tmp1d, val_addr4, vec_one),
                            ("&", vec_tmp1e, val_addr5, vec_one),
                            ("&", vec_tmp1f, val_addr6, vec_one),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("&", vec_tmp1g, val_addr7, vec_one),
                            ("&", vec_tmp1h, val_addr8, vec_one),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("+", vec_tmp2, vec_tmp1, vec_one),
                            ("+", vec_tmp2b, vec_tmp1b, vec_one),
                            ("+", vec_tmp2c, vec_tmp1c, vec_one),
                            ("+", vec_tmp2d, vec_tmp1d, vec_one),
                            ("+", vec_tmp2e, vec_tmp1e, vec_one),
                            ("+", vec_tmp2f, vec_tmp1f, vec_one),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("+", vec_tmp2g, vec_tmp1g, vec_one),
                            ("+", vec_tmp2h, vec_tmp1h, vec_one),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("*", idx_addr, idx_addr, vec_two),
                            ("*", idx_addr2, idx_addr2, vec_two),
                            ("*", idx_addr3, idx_addr3, vec_two),
                            ("*", idx_addr4, idx_addr4, vec_two),
                            ("*", idx_addr5, idx_addr5, vec_two),
                            ("*", idx_addr6, idx_addr6, vec_two),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("*", idx_addr7, idx_addr7, vec_two),
                            ("*", idx_addr8, idx_addr8, vec_two),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("+", idx_addr, idx_addr, vec_tmp2),
                            ("+", idx_addr2, idx_addr2, vec_tmp2b),
                            ("+", idx_addr3, idx_addr3, vec_tmp2c),
                            ("+", idx_addr4, idx_addr4, vec_tmp2d),
                            ("+", idx_addr5, idx_addr5, vec_tmp2e),
                            ("+", idx_addr6, idx_addr6, vec_tmp2f),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("+", idx_addr7, idx_addr7, vec_tmp2g),
                            ("+", idx_addr8, idx_addr8, vec_tmp2h),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("<", vec_tmp1, idx_addr, vec_n_nodes),
                            ("<", vec_tmp1b, idx_addr2, vec_n_nodes),
                            ("<", vec_tmp1c, idx_addr3, vec_n_nodes),
                            ("<", vec_tmp1d, idx_addr4, vec_n_nodes),
                            ("<", vec_tmp1e, idx_addr5, vec_n_nodes),
                            ("<", vec_tmp1f, idx_addr6, vec_n_nodes),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("<", vec_tmp1g, idx_addr7, vec_n_nodes),
                            ("<", vec_tmp1h, idx_addr8, vec_n_nodes),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("*", idx_addr, idx_addr, vec_tmp1),
                            ("*", idx_addr2, idx_addr2, vec_tmp1b),
                            ("*", idx_addr3, idx_addr3, vec_tmp1c),
                            ("*", idx_addr4, idx_addr4, vec_tmp1d),
                            ("*", idx_addr5, idx_addr5, vec_tmp1e),
                            ("*", idx_addr6, idx_addr6, vec_tmp1f),
                        ]
                    }
                )
                self.emit_bundle(
                    {
                        "valu": [
                            ("*", idx_addr7, idx_addr7, vec_tmp1g),
                            ("*", idx_addr8, idx_addr8, vec_tmp1h),
                        ]
                    }
                )

                if debug_round_writes:
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr2))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr3))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr4))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr5))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr6))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr7))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_idx_ptr, idx_addr8))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))

                    self.emit("store", ("vstore", debug_val_ptr, val_addr))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr2))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr3))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr4))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr5))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr6))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr7))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_addr8))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))

            for i in range(vec_unroll_end, vec_end, VLEN):
                emit_vec_block(i)
                if debug_round_writes:
                    self.emit("store", ("vstore", debug_idx_ptr, idx_buf + i))
                    self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, VLEN))
                    self.emit("store", ("vstore", debug_val_ptr, val_buf + i))
                    self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, VLEN))

            if fast_mode and tail_len:
                emit_vec_block(vec_end)

            if not fast_mode:
                for i in range(vec_end, batch_size):
                    i_const = self.scratch_const(i)
                    idx_addr = idx_buf + i
                    val_addr = val_buf + i
                    self.emit("alu", ("==", tmp1, cache_tag0, idx_addr))
                    self.emit("alu", ("==", tmp2, cache_tag1, idx_addr))
                    self.emit("alu", ("|", tmp3, tmp1, tmp2))
                    self.emit(
                        "flow", ("select", tmp_node_val, tmp1, cache_val0, cache_val1)
                    )
                    self.emit(
                        "alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_addr)
                    )
                    self.emit("load", ("load", tmp_miss_val, tmp_addr))
                    self.emit(
                        "flow",
                        ("select", tmp_node_val, tmp3, tmp_node_val, tmp_miss_val),
                    )
                    self.emit("alu", ("==", tmp2, tmp3, zero_const))
                    self.emit("alu", ("==", tmp1, cache_toggle, zero_const))
                    self.emit("alu", ("&", tmp1, tmp1, tmp2))
                    self.emit("alu", ("==", tmp3, cache_toggle, one_const))
                    self.emit("alu", ("&", tmp3, tmp3, tmp2))
                    self.emit("flow", ("select", cache_tag0, tmp1, idx_addr, cache_tag0))
                    self.emit(
                        "flow", ("select", cache_val0, tmp1, tmp_node_val, cache_val0)
                    )
                    self.emit("flow", ("select", cache_tag1, tmp3, idx_addr, cache_tag1))
                    self.emit(
                        "flow", ("select", cache_val1, tmp3, tmp_node_val, cache_val1)
                    )
                    self.emit("alu", ("^", tmp1, cache_toggle, one_const))
                    self.emit("flow", ("select", cache_toggle, tmp2, tmp1, cache_toggle))
                    self.emit("alu", ("^", val_addr, val_addr, tmp_node_val))
                    self.instrs.extend(self.build_hash(val_addr, tmp1, tmp2))
                    self.emit("alu", ("&", tmp1, val_addr, parity_mask))
                    self.emit("alu", ("+", tmp3, tmp1, one_const))
                    self.emit("alu", ("*", idx_addr, idx_addr, two_const))
                    self.emit("alu", ("+", idx_addr, idx_addr, tmp3))
                    self.emit("alu", ("<", tmp1, idx_addr, self.scratch["n_nodes"]))
                    self.emit("alu", ("*", idx_addr, idx_addr, tmp1))
                    if debug_round_writes:
                        self.emit("store", ("store", debug_idx_ptr, idx_addr))
                        self.emit("flow", ("add_imm", debug_idx_ptr, debug_idx_ptr, 1))
                        self.emit("store", ("store", debug_val_ptr, val_addr))
                        self.emit("flow", ("add_imm", debug_val_ptr, debug_val_ptr, 1))

        if not debug_round_writes:
            self.emit(
                "alu", ("+", idx_store_ptr, self.scratch["inp_indices_p"], zero_const)
            )
            self.emit(
                "alu", ("+", val_store_ptr, self.scratch["inp_values_p"], zero_const)
            )
            for i in range(0, vec_end, VLEN):
                idx_addr = idx_buf + i
                val_addr = val_buf + i
                self.emit("store", ("vstore", idx_store_ptr, idx_addr))
                self.emit("store", ("vstore", val_store_ptr, val_addr))
                self.emit("flow", ("add_imm", idx_store_ptr, idx_store_ptr, VLEN))
                self.emit("flow", ("add_imm", val_store_ptr, val_store_ptr, VLEN))

            for i in range(vec_end, batch_size):
                idx_addr = idx_buf + i
                val_addr = val_buf + i
                self.emit("store", ("store", idx_store_ptr, idx_addr))
                self.emit("store", ("store", val_store_ptr, val_addr))
                self.emit("flow", ("add_imm", idx_store_ptr, idx_store_ptr, 1))
                self.emit("flow", ("add_imm", val_store_ptr, val_store_ptr, 1))
        # Required to match with the yield in reference_kernel2
        if not fast_mode:
            self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(
        forest.height,
        len(forest.values),
        len(inp.indices),
        rounds,
        debug_round_writes=True,
        fast_mode=False,
    )
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
